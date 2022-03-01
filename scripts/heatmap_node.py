#!/usr/bin/env python3

import numpy as np
import math
import json
import argparse
import os
import ctypes

from cv_bridge import CvBridge, CvBridgeError
import cv2

import rospy
import message_filters
from edgetpu_ros_pose_estimation.msg import StringStamped
from person_msgs.msg import PersonStamped, PersonCovStamped, Person2DList
from sensor_msgs.msg import Image as ROS_Image
from sensor_msgs.msg import CameraInfo

import torch

_cam_info_topic = '/raw_output_info' # CameraInfo
_pose_2d_topic  = '/human_joints' # Person2DList
_heatmap_topic  = '/raw_output' # ROS_Image
_cam_id = 1

_img_size = (848, 480)
_last_stamp = None
_t0 = None

#_h36m_joints = {0: 'nose', 1: 'head', 2: 'neck', 3: 'belly', 4: 'root', 5: 'lsho', 6: 'rsho', 7: 'lelb', 8: 'relb', 9: 'lwri', 10: 'rwri', 11: 'lhip', 12: 'rhip', 13: 'lkne', 14: 'rkne', 15: 'lank', 16: 'rank'}

_simple_joints = {0: 'nose', 1: 'leye', 2: 'reye', 3: 'lear', 4: 'rear', 5: 'lsho', 6: 'rsho', 7: 'lelb', 8: 'relb', 9: 'lwri', 10: 'rwri', 11: 'lhip', 12: 'rhip', 13: 'lkne', 14: 'rkne', 15: 'lank', 16: 'rank'}

# --> output joints _simple_joints = {0: 'nose', 1: 'NECK', 2: 'ROOT', 3: 'lear', 4: 'rear', 5: 'lsho', 6: 'rsho', 7: 'lelb', 8: 'relb', 9: 'lwri', 10: 'rwri', 11: 'lhip', 12: 'rhip', 13: 'lkne', 14: 'rkne', 15: 'lank', 16: 'rank'}


_num_keypoints_simple = 17
_num_cameras = 1
_sequence = 0
#_heatmaps_base_path = '/home/sbultmann/datasets/hall/exported_data_single_person/heatmaps/'
#_json_file = None
_bridge = None

def get_gaussian_kernel_cov(size, x_sub, y_sub, cov_xx, cov_yy, cov_xy):
    tmp_size = 2 * size + 1
    gauss = np.zeros((tmp_size, tmp_size), dtype=np.float32)
    mux = size + x_sub
    muy = size + y_sub
    factor = -1. / (2*(cov_xx*cov_yy-cov_xy*cov_xy))
    x = np.arange(0, tmp_size, 1, np.float32)
    y = x[:, np.newaxis]
    return np.exp(factor * (cov_yy * (x-mux)**2 - 2*cov_xy * (x-mux) * (y-muy) + cov_xx * (y - muy)**2))

def mod_hm_channel(hm, kp, part_idx, crop_info):
    if kp.score == 0.0:
        return
    
    crop_w = crop_info[2]
    crop_h = crop_info[3]
    x0_coord = int(crop_info[0] - 0.5 * crop_w)
    y0_coord = int(crop_info[1] - 0.5 * crop_h)
    
    scale_x = float(crop_w) / hm.shape[1]
    scale_y = float(crop_h) / hm.shape[0]
    
    cov = np.array([kp.cov[0] / (scale_x * scale_x), kp.cov[1] / (scale_x * scale_y), kp.cov[1] / (scale_y * scale_x), kp.cov[2] / (scale_y*scale_y)]).reshape((2,2))
    cov *= 4.
    w, v = np.linalg.eig(cov)
    if abs(w[0]) > 1e3 or abs(w[1]) > 1e3:
        print('kp: {}, {}, score: {}'.format(kp.x, kp.y, kp.score))
        print('crop_w: {}, crop_h: {}, hm_shape: {}'.format(crop_w, crop_h, hm.shape))
        print('kp cov: {}, {}, {}'.format(kp.cov[0], kp.cov[1], kp.cov[2]))
        print('scale_x: {}, scale_y: {}'.format(scale_x, scale_y))
        print(cov)
        print(w)
    sigma = math.sqrt(max(w[0], w[1]))

    tmp_size = int(math.ceil(sigma * 3))
    
    mu_x = int((kp.x - x0_coord) / scale_x + 0.5)
    mu_y = int((kp.y - y0_coord) / scale_y + 0.5)
    
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= hm.shape[1] or ul[1] >= hm.shape[0] or br[0] < 0 or br[1] < 0:
        return
    
    sub_x = ((kp.x - x0_coord) / scale_x) - mu_x
    sub_y = ((kp.y - y0_coord) / scale_y) - mu_y
    g = get_gaussian_kernel_cov(tmp_size, sub_x, sub_y, cov[0,0], cov[1,1], cov[0,1]) * kp.score

    g_x = max(0, -ul[0]), min(br[0], hm.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], hm.shape[0]) - ul[1]
    img_x = max(0, ul[0]), min(br[0], hm.shape[1])
    img_y = max(0, ul[1]), min(br[1], hm.shape[0])

    hm[img_y[0]:img_y[1], img_x[0]:img_x[1], part_idx] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

def callback(msg_pose2d, msg_hm, msg_caminfo):
    global _last_stamp, _t0, _bridge, _sequence
    
    if _t0 is None:
        _t0 = msg_pose2d.header.stamp
    
    delta_t = (msg_pose2d.header.stamp - _t0).to_sec()
    time_idx = int(round(delta_t * 10))
        
    if _last_stamp is not None:
        if (msg_pose2d.header.stamp - _last_stamp).to_sec() > 0.300:
            # This triggeres if the camera did not observe any person for longer than 300ms.. (threshold can be adapted..)
            # The prediction sequence might need to be re-initialized in this case ?
            
            print('Cam {}: Next sequence, delta_t = {}s, timestep: {}.'.format(_cam_id, (msg_pose2d.header.stamp - _last_stamp).to_sec(), time_idx))
            if time_idx < 10:
                rospy.logwarn('Cam %d: Short sequence %d: only consists of %d steps!', _cam_id, _sequence, time_idx)
            
            _t0 = msg_pose2d.header.stamp
            delta_t = (msg_pose2d.header.stamp - _t0).to_sec()
            time_idx = int(round(delta_t * 10))
                        
            _sequence += 1
                        
        elif (msg_pose2d.header.stamp - _last_stamp).to_sec() > 0.120:
            n_missed_frames = int(round((msg_pose2d.header.stamp - _last_stamp).to_sec() * 10 - 1))
            rospy.logwarn('Cam %d: Missed %d frame(s): delta_t = %fs (should be 0.100s)', _cam_id, n_missed_frames, (msg_pose2d.header.stamp - _last_stamp).to_sec())
                
    _last_stamp = msg_pose2d.header.stamp
    
    
    pose2d = msg_pose2d.persons[0].keypoints # Keypoints are passed in absolute coordinates
    
    ## replace ch 1 with neck and ch 2 with root
    kpSh_left = pose2d[5]
    kpSh_right = pose2d[6]
    if kpSh_left.score > 0 and kpSh_right.score > 0:
        pose2d[1].x = (kpSh_left.x + kpSh_right.x) / 2.
        pose2d[1].y = (kpSh_left.y + kpSh_right.y) / 2.
        pose2d[1].score = (kpSh_left.score + kpSh_right.score) / 2.
        pose2d[1].cov = [(kpSh_left.cov[0] + kpSh_right.cov[0]) / 2., (kpSh_left.cov[1] + kpSh_right.cov[1]) / 2., (kpSh_left.cov[2] + kpSh_right.cov[2]) / 2.]
    else:
        rospy.logwarn('Cam %d: neck cannot be defined!', _cam_id)
        pose2d[1].x = 0.
        pose2d[1].y = 0.
        pose2d[1].score = 0.
        pose2d[1].cov = [0., 0., 0.]
    
    kpHp_left = pose2d[11]
    kpHp_right = pose2d[12]
    if kpHp_left.score > 0 and kpHp_right.score > 0:
        pose2d[2].x = (kpHp_left.x + kpHp_right.x) / 2.
        pose2d[2].y = (kpHp_left.y + kpHp_right.y) / 2.
        pose2d[2].score = (kpHp_left.score + kpHp_right.score) / 2.
        pose2d[2].cov = [(kpHp_left.cov[0] + kpHp_right.cov[0]) / 2., (kpHp_left.cov[1] + kpHp_right.cov[1]) / 2., (kpHp_left.cov[2] + kpHp_right.cov[2]) / 2.]
    else:
        rospy.logerr('Cam %d: root cannot be defined!', _cam_id)
        pose2d[2].x = 0.
        pose2d[2].y = 0.
        pose2d[2].score = 0.
        pose2d[2].cov = [0., 0., 0.]
    
    crop_info = [ctypes.c_int(msg_caminfo.roi.x_offset).value + 0.5 * msg_caminfo.roi.width, ctypes.c_int(msg_caminfo.roi.y_offset).value + 0.5 * msg_caminfo.roi.height, msg_caminfo.roi.width, msg_caminfo.roi.height] #x_center, y_center, width, height

    poses2d_list = [[kp.x, kp.y, kp.score] for kp in pose2d]

    try:
        hm = _bridge.imgmsg_to_cv2(msg_hm) # passthrough encoding
    except CvBridgeError as e:
        print(e)
    
    #replace ch 1 with neck and ch 2 with root
    hm_mod = hm.copy()
    hm_mod[:,:,1] = 0.
    hm_mod[:,:,2] = 0.
    mod_hm_channel(hm_mod, pose2d[1], 1, crop_info)
    mod_hm_channel(hm_mod, pose2d[2], 2, crop_info)
    # --> output joints {0: 'nose', 1: 'NECK', 2: 'ROOT', 3: 'lear', 4: 'rear', 5: 'lsho', 6: 'rsho', 7: 'lelb', 8: 'relb', 9: 'lwri', 10: 'rwri', 11: 'lhip', 12: 'rhip', 13: 'lkne', 14: 'rkne', 15: 'lank', 16: 'rank'}
    
    channel_mapping = [0, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    heatmap_input = hm_mod[:,:,channel_mapping]
    print('heatmap numpy: shape: {}, dtype: {}'.format(heatmap_input.shape, heatmap_input.dtype))
    #print(heatmap_input)
    
    input_tensor = torch.from_numpy(heatmap_input).cuda()
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0) # hwc -> chw and add batch dimension
    print('heatmap torch: shape: {}, dtype: {}'.format(input_tensor.shape, input_tensor.type()))
    #print(input_tensor)
    
    print('\n')
    
    ##TODO RUN INFERENCE...
    
    ##TODO Publish / visualize result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='camera number.')
    
    rospy.init_node('heatmap_prediction_node')
    
    args = parser.parse_args(rospy.myargv()[1:])
    
    print(torch.__version__)
    print('CUDA available: ' + str(torch.cuda.is_available()))
    print('cuDNN version: ' + str(torch.backends.cudnn.version()))
    
    global _cam_id
    _cam_id = args.camera
    
    global _bridge
    _bridge = CvBridge()
    
    caminfo_sub = message_filters.Subscriber('/d455' + _cam_info_topic, CameraInfo) # '/cam_{}'.format(_cam_id)
        
    pose_2d_sub = message_filters.Subscriber('/d455' + _pose_2d_topic, Person2DList) # '/cam_{}'.format(_cam_id)
        
    hm_sub = message_filters.Subscriber('/d455' + _heatmap_topic, ROS_Image, buff_size = 16777216) # '/cam_{}'.format(_cam_id) + _heatmap_topic
        
    ts = message_filters.TimeSynchronizer([pose_2d_sub, hm_sub, caminfo_sub], 10)
    ts.registerCallback(callback)
    
    rospy.spin()

if __name__ == '__main__':
    main()

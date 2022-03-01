#!/usr/bin/env python3

import argparse
import time
import os

from PIL import Image
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from scipy.optimize import linear_sum_assignment
from multiprocessing import Process, Value, Array, Lock, Condition, Pipe
import ctypes

import tensorrt as trt
import utils.inference as inference_utils # TRT inference wrapper
import utils.model as model_utils
from utils.paths import PATHS # Path management

from estimator import PoseEstimator, Human, BodyPart, BBox, data_associate
from common import CocoColors_inv

import rospy
from sensor_msgs.msg import Image as ROS_Image
from person_msgs.msg import Person2DList, Person2D, Keypoint2D

from my_edgetpuvision import utils as edgetpu_utils

# Model used for inference
MODEL_NAME = 'person_ssd_480p'
MODEL_NAME_POSE = 'person_pose_192x256'

# Precision command line argument -> TRT Engine datatype
TRT_PRECISION_TO_DATATYPE = {
    16: trt.DataType.HALF,
    32: trt.DataType.FLOAT
}

# Layout of TensorRT network output metadata
TRT_PREDICTION_LAYOUT = {
    "image_id": 0,
    "label": 1,
    "confidence": 2,
    "xmin": 3,
    "ymin": 4,
    "xmax": 5,
    "ymax": 6
}

TRT_HEATMAP_SHAPE = (64, 48, 17)
TRT_HEATMAP_SIZE  = 64 * 48 * 17

def print_results(inference_rate, humans):
    print('\nInference (rate=%.2f fps):' % inference_rate)
    print('Num person: %d' % len(humans))
    for i, human in enumerate(humans):
        print('    %d: %s.' % (i, human))

def fetch_prediction_field(field_name, detection_out, pred_start_idx):
    return detection_out[pred_start_idx + TRT_PREDICTION_LAYOUT[field_name]]

def convert(det, size, pred_start_idx, pred_thresh):
    #image_id = int(fetch_prediction_field("image_id", det, pred_start_idx))
    #label = int(fetch_prediction_field("label", det, pred_start_idx)) # only one classe: persons
    confidence = fetch_prediction_field("confidence", det, pred_start_idx)
    if confidence > pred_thresh:
        x0 = fetch_prediction_field("xmin", det, pred_start_idx)
        y0 = fetch_prediction_field("ymin", det, pred_start_idx)
        x1 = fetch_prediction_field("xmax", det, pred_start_idx)
        y1 = fetch_prediction_field("ymax", det, pred_start_idx)

        bbox = [x0 * size[0], y0 * size[1], (x1 - x0) * size[0], (y1 - y0) * size[1]] # x, y, w, h
        bbox_orig = np.array([x0 * size[0], y0 * size[1], x1 * size[0], y1 * size[1]]) # x0, y0, x1, y1, abs. coords.
        return dict(bbox=bbox, bbox_orig=bbox_orig, score=confidence)
    else:
        return None

def convert_prev(crop, factor):
    x0, y0, x1, y1 = crop
    w = x1 - x0
    h = y1 - y0
    tmp_factor = (factor - 1.) / 2.
    bbox = [x0 - tmp_factor * w, y0 - tmp_factor * h, factor * w, factor * h] # x, y, w, h
    return dict(bbox=bbox, bbox_orig=crop, score=1.0)

def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(boxA[0], boxB[0]) #x0
    y1 = max(boxA[1], boxB[1]) #y0
    x2 = min(boxA[2], boxB[2]) #x1
    y2 = min(boxA[3], boxB[3]) #y1

    interArea = max(0, x2 - x1) * max(0, y2 - y1)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / (float(areaA + areaB - interArea) + np.spacing(1))

    return iou

def associate_feedback(crop_data_det, people_fb):
    C = np.ones((len(crop_data_det), len(people_fb)))
    for i, crop in enumerate(crop_data_det):
        bbox_det = crop['bbox_orig']
        for j, human_fb in enumerate(people_fb):
            bbox_fb = human_fb['bbox']
            iou = bb_iou(bbox_det, bbox_fb)
            #print('iou det {}, fb {}: {}'.format(i, j, iou))
            C[i, j] = 1.0 - iou

    row_ind, col_ind = linear_sum_assignment(C)
    col_ind_out = -np.ones(len(crop_data_det), dtype=np.int32)
    for idx in range(len(row_ind)):
        if C[row_ind[idx], col_ind[idx]] < 0.7: # upper bound for total cost (min iou = 0.3)
            col_ind_out[row_ind[idx]] = col_ind[idx]

    #print('final association: {}'.format(col_ind_out))
    return col_ind_out

def nms_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    return boxes[pick]

def crop_bbox(img, d, input_shape):
    bbox = np.array(d['bbox']).astype(np.float32)
    
    x, y, w, h = bbox
    aspect_ratio = input_shape[1] / input_shape[0]
    center = np.array([x + w * 0.5, y + h * 0.5])
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w,h]) * 1.25
    input_size = (input_shape[1], input_shape[0])
    transform_data = (scale[0] / input_size[0], 0., center[0] - 0.5 * scale[0], 0., scale[1] / input_size[1], center[1] - 0.5 * scale[1])
    #print(transform_data)
    
    cropped_img = img.transform(input_size, Image.AFFINE, data=transform_data, resample=Image.BILINEAR)
    crop_info = np.asarray([center[0]-scale[0]*0.5, center[1]-scale[1]*0.5, center[0]+scale[0]*0.5, center[1]+scale[1]*0.5]) # x0, y0, x1, y1
    
    return [cropped_img, crop_info]

def humans_to_msg(humans):
    msg = Person2DList()
    
    for human in humans:
        human_msg = Person2D()
        human_msg.score = human.score
        human_msg.bbox = human.get_bbox()
        
        for i in range(TRT_HEATMAP_SHAPE[2]):
            kp = Keypoint2D()
            if i in human.body_parts.keys():
                body_part = human.body_parts[i]
                kp.x = body_part.x
                kp.y = body_part.y
                kp.score = body_part.score
                #kp.cov = TODO
                
            human_msg.keypoints.append(kp)
        
        msg.persons.append(human_msg)
        
    return msg

class pose_estimator: 
    def __init__(self, args):
        self.camera = args.camera
        self.print = args.print
        self.pub_overlay = args.pub_overlay # TODO: keep this ?
        self.flip = args.flip
        self.det_thresh = args.det_thresh
        self.top_k = args.top_k
        self.part_thresh = args.part_thresh
        self.feedback_type = args.feedback_type
        self.debug_heatmap = args.debug_heatmap
        self.max_batch_size_pose = args.max_batch_size_pose
        
        self.num_keypoints = 17
        self.n_det = 20 # run detector every n_det frames
        self.prev_crop_factor = 1.25
        self.nms_thresh = 0.40
        self.last_t = 0
        
        ssd_model_uff_path = PATHS.get_model_uff_path(MODEL_NAME)
        pose_model_uff_path = PATHS.get_model_pose_uff_path(MODEL_NAME_POSE)
        self.det_input_width = model_utils.ModelData.get_input_width()
        self.det_input_height = model_utils.ModelData.get_input_height()
        self.input_shape_pose = (model_utils.ModelDataPose.get_input_height(), model_utils.ModelDataPose.get_input_width())
        self.prediction_fields = len(TRT_PREDICTION_LAYOUT)

        # Set up all TensorRT data structures needed for inference
        self.trt_inference_wrapper = inference_utils.TRTInference(
            args.trt_engine_path, ssd_model_uff_path,
            trt_engine_datatype=args.trt_engine_datatype,
            batch_size=1)
        
        self.trt_inference_wrapper_pose = inference_utils.TRTInference(
            args.trt_engine_path_pose, pose_model_uff_path,
            trt_engine_datatype=args.trt_engine_datatype,
            batch_size=args.max_batch_size_pose, pose=True)
        
        detection_out, keep_count_out = self.trt_inference_wrapper.infer(np.zeros((3, 480, 640))) # test...
        print('TEST detector inference done')
        
        self.fps_counter  = edgetpu_utils.avg_fps_counter(30)
        self.frame_cnt = 0
        self.prev_crop = np.array([])
        self.feedback_delay_cum = 0.
        self.feedback_cnt = 0
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/{}/color/image_raw'.format(self.camera), ROS_Image, self.inference_cb,  queue_size = 1, buff_size = 16777216)
        self.publisher_humans = rospy.Publisher('/{}/human_joints'.format(self.camera), Person2DList, queue_size=1)
    
        if self.pub_overlay:
            self.publisher_image_overlay = rospy.Publisher('/{}/image_overlay'.format(self.camera), ROS_Image, queue_size=1)
            
        if self.debug_heatmap:
            self.heatmap_buffer = Array(ctypes.c_uint8, int(48 * 64 * 3)) # rgb-color image of heatmap
            self.heatmap_timestamp = Value(ctypes.c_double, 0.)
            self.heatmap_updated = Value(ctypes.c_bool, False)
            self.publisher_heatmap = rospy.Publisher('/{}/heatmap'.format(self.camera), ROS_Image, queue_size=1)
            self.debug_hm_conn_rcv, self.debug_hm_conn_send = Pipe(duplex = False) 
            self.debug_heatmap_thread = Process(target = self.debug_heatmap_cb)
            self.debug_heatmap_thread.start()
            
        self.max_fb_delay = 1.
        self.w_feedback = Value(ctypes.c_double, 1.)
        self.sigma = 1.5
        self.sigma_score_factor = 7.5
        self.sigma_max = 10.
        self.min_dist_swap = [3, 6]
        self.max_env_swap = 2
        if 'hm' in self.feedback_type or 'heatmap' in self.feedback_type:
            self.feedback_type = 'none'
            print('feedback type \"heatmap\" is not supported. not using any feedback.')
            
        elif 'skel' in self.feedback_type:
            self.feedback_type = 'skeleton'
            self.feedback_skeleton = Array(ctypes.c_char, self._MAX_JSON_STR_LEN)
            self.feedback_skel_stamp = Value(ctypes.c_double, 0.)
            self.feedback_skeleton_sub = rospy.Subscriber('/{}/skel_pred'.format(self.camera), StringStamped, self.feedback_skel_callback,  queue_size = 1)
            self.fb_skel_conn_main, self.fb_skel_conn_cb = Pipe()
            self.render_fb_hm_thread = Process(target = self.render_fb_heatmap_cb)
            self.render_fb_hm_thread.start()
            print('Using feedback type \"skeleton\"')
        else:
            print('Not using any feedback')
            
    def feedback_skel_callback(self, skel_msg):
        with self.feedback_skel_stamp.get_lock():
            self.feedback_skeleton.value = skel_msg.data.encode()
            self.feedback_skel_stamp.value = skel_msg.header.stamp.to_sec()

    def feedback_hm_callback(self, img_msg):
        with self.feedback_hm_stamp.get_lock():
            hm_np = np.frombuffer(self.feedback_heatmaps.get_obj(), np.uint8)
            np.copyto(hm_np, img_msg.data)
            self.feedback_hm_stamp.value = img_msg.header.stamp.to_sec()
            
    def debug_heatmap_cb(self):
        while not rospy.is_shutdown():
            heatmaps, heatmaps_feedback, ts_sec = self.debug_hm_conn_rcv.recv()

            heatmaps_color = np.zeros((heatmaps.shape[0], heatmaps.shape[1], 3))
            for ch_idx in range(heatmaps.shape[-1]):
                heatmaps_color[:,:,0] = np.maximum(heatmaps_color[:,:,0], 255 * heatmaps[:,:,ch_idx])
                heatmaps_color[:,:,1] = np.maximum(heatmaps_color[:,:,1], 255 * heatmaps[:,:,ch_idx])
                heatmaps_color[:,:,2] = np.maximum(heatmaps_color[:,:,2], 255 * heatmaps[:,:,ch_idx])

            if heatmaps_feedback is not None:
                for ch_idx in range(heatmaps_feedback.shape[-1]):
                    heatmaps_color[:,:,0] = np.maximum(heatmaps_color[:,:,0], CocoColors_inv[ch_idx][0] * heatmaps_feedback[:,:,ch_idx])
                    heatmaps_color[:,:,1] = np.maximum(heatmaps_color[:,:,1], CocoColors_inv[ch_idx][1] * heatmaps_feedback[:,:,ch_idx])
                    heatmaps_color[:,:,2] = np.maximum(heatmaps_color[:,:,2], CocoColors_inv[ch_idx][2] * heatmaps_feedback[:,:,ch_idx])

            if heatmaps_color.max() > 0:
                heatmaps_color -= heatmaps_color.min()
                heatmaps_color *= 255.0/heatmaps_color.max()
                with self.heatmap_updated.get_lock():
                    heatmap_np = np.frombuffer(self.heatmap_buffer.get_obj(), np.uint8)
                    np.copyto(heatmap_np, (255. - heatmaps_color).astype(np.uint8).flatten())
                    self.heatmap_timestamp.value = ts_sec
                    self.heatmap_updated.value = True

    def render_fb_heatmap_cb(self):
        hm_shape = self.fb_skel_conn_cb.recv()
        print('received hm_shape: {}'.format(hm_shape))

        while not rospy.is_shutdown():
            human_fb, crop_info = self.fb_skel_conn_cb.recv()

            if human_fb is not None:
                heatmaps_feedback, coords_crop = self.generate_heatmap(human_fb, self.orig_size, hm_shape, crop_info)
                self.fb_skel_conn_cb.send([heatmaps_feedback, coords_crop])
            else:
                self.fb_skel_conn_cb.send([None, None])

    def generate_heatmap(self, human_json, imsize, hm_shape, crop_info):
        hm = np.zeros(hm_shape, dtype=np.float32) 
        
        crop_w = crop_info[2] - crop_info[0]
        crop_h = crop_info[3] - crop_info[1]
        x0_coord = int(crop_info[0])
        y0_coord = int(crop_info[1])
        coords_crop = []
        
        if crop_w == 0 or crop_h == 0:
            return hm, coords_crop

        for part_idx in range(hm_shape[-1]):
            if human_json['keypoints'][part_idx] is None:
                coords_crop.append([0, 0, 0])
                continue
            
            kp = human_json['keypoints'][part_idx] # [x, y, sigma]
            sigma = min(self.sigma_max, self.sigma + self.sigma_score_factor * kp[2])
            tmp_size = sigma * 3
            tmp_factor = self.sigma / sigma
            
            x_orig = kp[0] * imsize[0]
            y_orig = kp[1] * imsize[1]
            mu_x = int((x_orig - x0_coord) / float(crop_w) * hm.shape[1] + 0.5)
            mu_y = int((y_orig - y0_coord) / float(crop_h) * hm.shape[0] + 0.5)
            coords_crop.append([mu_x, mu_y, tmp_size])
            
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= hm.shape[1] or ul[1] >= hm.shape[0] or br[0] < 0 or br[1] < 0:
                continue
            
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) * tmp_factor

            g_x = max(0, -ul[0]), min(br[0], hm.shape[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], hm.shape[0]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], hm.shape[1])
            img_y = max(0, ul[1]), min(br[1], hm.shape[0])

            hm[img_y[0]:img_y[1], img_x[0]:img_x[1], part_idx] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
            
        return hm, coords_crop
    
    def swap_hm_channels(self, coords, hm, symm_pairs):
        for pair in symm_pairs:
            p0 = coords[pair[0]]
            p1 = coords[pair[1]]
            dist = np.linalg.norm([p0[0]-p1[0], p0[1]-p1[1]])
            min_dist = self.min_dist_swap[0] if pair[0] <= 9 else self.min_dist_swap[1]
            env_size = min(self.max_env_swap, (dist - min_dist) / 2.)
            if env_size < 0:
                #rospy.logwarn('feedback points p{}: {} and p{}: {} too close: {:.2f}px. Not checking for swap!'.format(pair[0], p0, pair[1], p1, dist))
                continue
            
            env_size = int(env_size)
            ul0 = [int(p0[0] - env_size), int(p0[1] - env_size)]
            br0 = [int(p0[0] + env_size + 1), int(p0[1] + env_size + 1)]
            ul1 = [int(p1[0] - env_size), int(p1[1] - env_size)]
            br1 = [int(p1[0] + env_size + 1), int(p1[1] + env_size + 1)]
            
            if ul0[0] >= hm.shape[1] or ul0[1] >= hm.shape[0] or br0[0] < 0 or br0[1] < 0 or ul1[0] >= hm.shape[1] or ul1[1] >= hm.shape[0] or br1[0] < 0 or br1[1] < 0:
                #rospy.logwarn('feedback points p{}: {} or p{}: {} outside crop!'.format(pair[0], p0, pair[1], p1))
                continue
            
            hm_x0 = max(0, ul0[0]), min(br0[0], hm.shape[1])
            hm_y0 = max(0, ul0[1]), min(br0[1], hm.shape[0])
            sum00 = np.sum(hm[hm_y0[0]:hm_y0[1], hm_x0[0]:hm_x0[1], pair[0]])
            sum01 = np.sum(hm[hm_y0[0]:hm_y0[1], hm_x0[0]:hm_x0[1], pair[1]])
            swap0 = sum00 < sum01
            #print('  part: {}. hm-left: {}, hm-right: {}'.format(pair[0], sum00, sum01))
            
            hm_x1 = max(0, ul1[0]), min(br1[0], hm.shape[1])
            hm_y1 = max(0, ul1[1]), min(br1[1], hm.shape[0])
            sum11 = np.sum(hm[hm_y1[0]:hm_y1[1], hm_x1[0]:hm_x1[1], pair[1]])
            sum10 = np.sum(hm[hm_y1[0]:hm_y1[1], hm_x1[0]:hm_x1[1], pair[0]])
            swap1 = sum11 < sum10
            #print('  part: {}. hm-right: {}, hm-left: {}'.format(pair[1], sum11, sum10))
            
            if swap0 and swap1:
                print('Left-Right swap detected! joints {} <-> {}. exchanging heatmap channels'.format(pair[0], pair[1]))
                tmp = hm[:,:,pair[0]].copy()
                hm[:,:,pair[0]] = hm[:,:,pair[1]]
                hm[:,:,pair[1]] = tmp
            
    def inference_cb(self, img_msg):
        try:
            orig_image_np = self.bridge.imgmsg_to_cv2(img_msg, "rgb8")
        except CvBridgeError as e:
            print(e)
            
        if(self.flip):
                orig_image_np = cv2.flip(orig_image_np, -1)
                
        ts_sec = img_msg.header.stamp.to_sec()
        
        inference_rate = next(self.fps_counter)
        start = time.monotonic()
        
        if self.frame_cnt % self.n_det == 0 or len(self.prev_crop) == 0:
            if orig_image_np.shape[0] != self.det_input_height or orig_image_np.shape[1] != self.det_input_width:
                img_resized = cv2.resize(orig_image_np, (self.det_input_width, self.det_input_height))
            else:
                img_resized = orig_image_np
                
            tensor = img_resized.transpose((2, 0, 1)) # HWC -> CHW
            tensor = (2.0 / 255.0) * tensor - 1.0 # Normalize to [-1.0, 1.0] interval (expected by model)
            
            start = time.monotonic()
            detection_out, keep_count_out = self.trt_inference_wrapper.infer(tensor) # Detector expects rgb input

            crop_data_raw = [convert(detection_out, (orig_image_np.shape[1], orig_image_np.shape[0]), det * self.prediction_fields, self.det_thresh) for det in range(int(keep_count_out[0]))]
            crop_data = [x for x in crop_data_raw if x is not None]
            #print(crop_data)
        else:
            prev_crop_nms = nms_fast(self.prev_crop, self.nms_thresh)
            crop_data = [convert_prev(crop, self.prev_crop_factor) for crop in prev_crop_nms]
        det_time = time.monotonic() - start

        humans = []
        crop_times = []
        crop_infos = []
        pp_times = []
        t_feedback = None
        skel_feedback_json = None
        #w_fb = self.w_feedback.value
        
        if len(crop_data) > 0:
            orig_img = Image.fromarray(orig_image_np,'RGB')
             
            #TODO: feedaback rendering currently not implemented in batch-mode
            #if self.feedback_type == 'skeleton' and w_fb > 0.0:
                #with self.feedback_skel_stamp.get_lock():
                    #if self.feedback_skel_stamp.value > 0.:
                        #skel_feedback = self.feedback_skeleton.value.decode()
                        #t_feedback = self.feedback_skel_stamp.value
                #if t_feedback is not None and abs(t_feedback - ts_sec) < self.max_fb_delay:
                    #skel_feedback_json = json.loads(skel_feedback)
                    #fb_association = associate_feedback(crop_data, skel_feedback_json['people'])

            #else:
                #w_fb = 0.0
        
        for idx in range(min(len(crop_data), self.max_batch_size_pose)): # TODO: adapt max-batch-size to number of detections...
            start_crop = time.monotonic()
            cropped_img, crop_info = crop_bbox(orig_img, crop_data[idx], self.input_shape_pose)
            img_np = (2.0 / 255.0) * np.asarray(cropped_img)[:,:,::-1].transpose((2,0,1)) - 1.0 # reverse color channesl (rgb -> bgr)), HWC->CHW, and normalize
            self.trt_inference_wrapper_pose.numpy_array[idx] = img_np.ravel()
            crop_infos.append(crop_info)
            crop_times.append(time.monotonic() - start_crop)
            
            #TODO: feedaback rendering currently not implemented in batch-mode
            #use_skel_feedback = False
            #if skel_feedback_json is not None and abs(t_feedback - ts_sec) < self.max_fb_delay:
                #self.fb_skel_conn_main.send([skel_feedback_json['people'][int(fb_association[i])] if fb_association[i] >= 0 else None, crop_info])
                ##print('feedback skelton sent! Index {}, Crop_info: {}'.format(fb_association[i], crop_info))
                #use_skel_feedback = True

        start_inf = time.monotonic()
        heatmaps_raw = self.trt_inference_wrapper_pose.infer_batch_pose()[0]
        inference_time = (time.monotonic() - start_inf)
        
        hm_offset = 0
        for idx in range(min(len(crop_data), self.max_batch_size_pose)):
            start_pp = time.monotonic()
            
            hm = heatmaps_raw[hm_offset : hm_offset + TRT_HEATMAP_SIZE].reshape(TRT_HEATMAP_SHAPE) / 255.
            hm_offset += TRT_HEATMAP_SIZE
            human = PoseEstimator.estimate_single(hm, part_threshold = self.part_thresh, score_thresh = 0., crop_info = crop_infos[idx])
            
            #TODO: feedaback rendering currently not implemented in batch-mode
            #if use_skel_feedback:
                #delta_t = t_feedback - ts_sec
                #self.feedback_delay_cum += delta_t
                #self.feedback_cnt += 1
                #if abs(delta_t) > 0.25:
                    #print('Feedback skeleton has delay larger than 0.25s: delta_t: {:.2f}s.'.format(delta_t))
                
                #heatmaps_feedback, coords_crop = self.fb_skel_conn_main.recv()
                
                #if heatmaps_feedback is not None:
                    ##if len(coords_crop) == self.num_keypoints:
                        ##self.swap_hm_channels(coords_crop, heatmaps, engine_pose.kps_symmetry)
                    
                    #heatmaps_fused = heatmaps + w_fb * heatmaps_feedback
                
                    #human = PoseEstimator.estimate_single(heatmaps_fused, part_threshold = self.part_thresh, score_thresh = 0., crop_info = crop_info, heat_mat_orig = heatmaps)

                #else:
                    #human = PoseEstimator.estimate_single(heatmaps, part_threshold = self.part_thresh, score_thresh = 0., crop_info = crop_info)

            #else:
                #human = PoseEstimator.estimate_single(heatmaps, part_threshold = self.part_thresh, score_thresh = 0., crop_info = crop_info)
                
                #if (self.feedback_type == 'heatmap' or self.feedback_type == 'skeleton') and w_fb > 0.0:
                    #print('No feedback of type \"{}\" received! delta_t: {}'.format(self.feedback_type, t_feedback - ts_sec if t_feedback is not None else 'NaN'))
                
            pp_times.append(time.monotonic() - start_pp)
            
            if len(human) > 0 and len(human[0].body_parts) > 0:
                #human[0].score /= (1. + w_fb)
                if human[0].score > self.part_thresh:
                    humans.append(human[0])
            
        self.prev_crop = np.array([human.get_bbox() for human in humans])
        
        humans_msg = humans_to_msg(humans)
        humans_msg.header = img_msg.header
        self.publisher_humans.publish(humans_msg)

        if self.debug_heatmap and len(crop_data) > 0: # render and publish debug heatmap
            self.debug_hm_conn_send.send([hm, None, ts_sec]) # heatmaps_feedback if use_skel_feedback else

        if self.print:
            print_results(inference_rate, humans)

        if self.pub_overlay and len(humans) > 0:
            img_overlay = np.asarray(PoseEstimator.draw_humans_abs(orig_img, humans))
            img_msg_overlay = self.bridge.cv2_to_imgmsg(img_overlay, "rgb8")
            img_msg_overlay.header = img_msg.header
            self.publisher_image_overlay.publish(img_msg_overlay)
        
        elapsed = time.monotonic() - start
        
        self.frame_cnt += 1
        
        if len(crop_times) > 0:
            print('Inference rate={:.2f} fps, det time: {:.2f}ms ({:02d} dets), crop time (avg): {:.2f}ms, inference time: {:.2f}ms, pp time (avg): {:.2f}ms, total time: {:.2f}ms, fb delay (avg): {:.2f}ms.\r'.format(inference_rate, det_time * 1000, len(crop_times), (sum(crop_times) / len(crop_times)) * 1000, inference_time * 1000, (sum(pp_times) / len(pp_times)) * 1000, elapsed * 1000, 1000 * self.feedback_delay_cum / self.feedback_cnt if self.feedback_cnt > 0 else 0.), end="")
        else:
            print('Inference rate={:.2f} fps, det time: {:.2f}ms (0 dets), total time: {:.2f}ms, fb delay (avg): {:.2f}ms.\r'.format(inference_rate, det_time * 1000, elapsed * 1000, 1000 * self.feedback_delay_cum / self.feedback_cnt if self.feedback_cnt > 0 else 0.), end="")
            
    def publish_heatmap(self):
        heatmap_img = None
        ts_sec = None
        with self.heatmap_updated.get_lock():
            if self.heatmap_updated.value:
                heatmap_img = np.frombuffer(self.heatmap_buffer.get_obj(), np.uint8).copy()
                ts_sec = self.heatmap_timestamp.value
                self.heatmap_updated.value = False
                
        if heatmap_img is not None:
            heatmap_img = heatmap_img.reshape((64, 48, 3))
            img_msg_heatmap = ROS_Image()
            img_msg_heatmap.height = heatmap_img.shape[0]
            img_msg_heatmap.width = heatmap_img.shape[1]
            img_msg_heatmap.encoding = "rgb8"
            img_msg_heatmap.data = heatmap_img.tostring()
            img_msg_heatmap.step = int(len(img_msg_heatmap.data) / img_msg_heatmap.height)
            img_msg_heatmap.header.stamp = rospy.Time.from_sec(ts_sec)
            self.publisher_heatmap.publish(img_msg_heatmap)

    def spin(self):
        #i = 0
        r = rospy.Rate(30) # 30hz
        while not rospy.is_shutdown():
            
            if self.debug_heatmap and self.publisher_heatmap.get_num_connections() > 0:
                self.publish_heatmap()

            #if i % 100 == 0):
                #w_fb = float(rospy.get_param('/heatmap_feedback_weight', 1.))
                #with self.w_feedback.get_lock():
                    #if w_fb != self.w_feedback.value:
                        #self.w_feedback.value = w_fb
                        #print('w_feedback updated to {}.    '.format(self.w_feedback.value))
            #i += 1
            r.sleep()

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', help='camera name and prefix', required=True)
    parser.add_argument('--det_thresh', type=float, default = 0.50, help = 'detector threshold')
    parser.add_argument('--top_k', type=int, default = 10, help = 'max. number of people to track')
    parser.add_argument('--part_thresh', type=float, default = 0.30, help = 'pose keypoint confidence threshold')
    parser.add_argument('--feedback_type', type=str, default='none', help='type of feedback to receive from fusion backend.')
    parser.add_argument('--print', default=False, action='store_true', help='Print inference results')
    parser.add_argument('--pub_overlay', default=False, action='store_true', help='Publish the overlayed image')
    parser.add_argument('--debug_heatmap', default=False, action='store_true', help='Render and publish debug heatmap (det + fb)')
    parser.add_argument('--flip', default=False, action='store_true', help='flip the camera image')
    parser.add_argument('-p', '--precision', type=int, choices=[32, 16], default=16, help='desired TensorRT float precision to build an engine with')
    parser.add_argument('-b', '--max_batch_size_pose', type=int, default=5, help='max TensorRT pose-crop batch size')
    
    rospy.init_node('pose_estimation_node_jetson_trt')
    
    args = parser.parse_args(rospy.myargv()[1:])
    
    # Fetch TensorRT engine path and datatype
    args.trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[args.precision]
    args.trt_engine_path = PATHS.get_engine_path(args.trt_engine_datatype, 1)
    args.trt_engine_path_pose = PATHS.get_engine_pose_path(args.trt_engine_datatype, args.max_batch_size_pose)
    
    pose_est = pose_estimator(args)
    
    pose_est.w_feedback.value = float(rospy.get_param('/heatmap_feedback_weight', 1.))
    rospy.loginfo('Using feedback weight: {}'.format(pose_est.w_feedback.value))

    pose_est.spin()
    
    pose_est.trt_inference_wrapper.destory();
    pose_est.trt_inference_wrapper_pose.destory();
    
    if pose_est.debug_heatmap:
        pose_est.debug_heatmap_thread.join(timeout=1.0)
        if(pose_est.debug_heatmap_thread.is_alive()):
            print('\nDebug Heatmap Thread did not exit cleanly. terminating.')
            pose_est.debug_heatmap_thread.kill()
        
    if pose_est.feedback_type == 'skeleton':
        pose_est.render_fb_hm_thread.join(timeout=1.0)
        if(pose_est.render_fb_hm_thread.is_alive()):
            print('\nFeedback heatmap render thread did not exit cleanly. terminating.')
            pose_est.render_fb_hm_thread.kill()

if __name__ == '__main__':
    main()

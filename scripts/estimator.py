import math
import json
import datetime
import collections
import numpy as np
from scipy import optimize

import common
from common import CocoPart

#from pafprocess import pafprocess

from PIL import Image, ImageDraw, ImageFont

#import time

def _round(v):
    return int(round(v))


def _include_part(part_list, part_idx):
    for part in part_list:
        if part_idx == part.part_idx:
            return True, part
    return False, None

def TimestampMillisec64():
        return int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)

BBox = collections.namedtuple('BBox', ('x1', 'y1', 'x2', 'y2'))
BBox.area = lambda self: (self.x2 - self.x1) * (self.y2 - self.y1)
BBox.scale = lambda self, sx, sy: BBox(x1=self.x1 * sx, y1=self.y1 * sy, x2=self.x2 * sx, y2=self.y2 * sy)
BBox.__str__ = lambda self: 'BBox(x1=%.2f y1=%.2f x2=%.2f y2=%.2f)' % self

def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(boxA.x1, boxB.x1)
    y1 = max(boxA.y1, boxB.y1)
    x2 = min(boxA.x2, boxB.x2)
    y2 = min(boxA.y2, boxB.y2)

    #print('IOU: BBox1: {}, BBox2: {}'.format(boxA, boxB))
                     
    # compute the area of intersection rectangle
    interArea = max(0, x2 - x1) * max(0, y2 - y1)
                             
    # compute the IOU by taking the intersection area and dividing it by the sum of both BBox  areas - the interesection area
    iou = interArea / (float(boxA.area() + boxB.area() - interArea) + np.spacing(1))
                                                             
    # return the intersection over union value
    return iou

def data_associate(humans, prev_humans, w_iou = 0.5, min_iou = 0.2, min_oks = 0.3, sigmas = None):

    w_oks = 1.0 - w_iou
    max_costs = 1.0 - (w_iou * min_iou + w_oks * min_oks)

    _min_num_joints = 5
    _min_num_obs = 3

    C = np.ones((len(humans), len(prev_humans)))
    for i, human in enumerate(humans):
        for j, prev_human in enumerate(prev_humans):
            iou = bb_iou(human.get_bbox(), prev_human.get_bbox())
            oks = human.oks_dist(prev_human, sigmas = sigmas)
            C[i, j] = 1.0 - (w_iou * iou + w_oks * oks)
            #print('Assign i={} to j={}: iou: {}, oks: {}, C[i,j]: {}.'.format(i, j, iou, oks, C[i,j]))

    row_ind, col_ind = optimize.linear_sum_assignment(C)
    #print('assignment: current humans {} to previous tracks {}.'.format(row_ind, col_ind))

    obs_rejected = False
    for idx in range(len(row_ind)):
        if C[row_ind[idx], col_ind[idx]] < max_costs: # upper bound for total cost
            humans[row_ind[idx]].id = prev_humans[col_ind[idx]].id # save associated id.
            humans[row_ind[idx]].num_obs = prev_humans[col_ind[idx]].num_obs + 1
            #print('Human ID: {}, Num observations: {}.'.format(humans[row_ind[idx]].id, humans[row_ind[idx]].num_obs))
        else:
            col_ind[idx] = -1 # reset col idx so that the temporary occlusion mechanism is used in this case.
            obs_rejected = True

    if len(prev_humans) > len(humans) or obs_rejected: # a previously tracked human is no longer observed. temporary occlusion ?
        for idx in range(len(prev_humans)):
            if not idx in col_ind: # this human has no current observation associated
                if prev_humans[idx].num_obs > _min_num_obs and len(prev_humans[idx].body_parts) > _min_num_joints:
                    dummy_human = Human([])
                    dummy_human.id = prev_humans[idx].id
                    dummy_human.num_obs = prev_humans[idx].num_obs
                    humans.append(dummy_human)
                    #print('adding dummy human for id: {}.'.format(dummy_human.id))

class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'body_parts_xyc', 'pairs', 'uidx_list', 'score', 'id', 'num_obs')

    kpt_oks_sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35]) / 10.0 # For Openpose keypoints!

    def __init__(self, pairs):
        self.id = -1
        self.num_obs = 0
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        self.body_parts_xyc = {}
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def get_bbox(self):
        if len(self.body_parts) > 0:
            max_x = self.body_parts_xyc[max(self.body_parts_xyc, key=lambda key: self.body_parts_xyc[key][0])][0]
            min_x = self.body_parts_xyc[min(self.body_parts_xyc, key=lambda key: self.body_parts_xyc[key][0])][0]
            max_y = self.body_parts_xyc[max(self.body_parts_xyc, key=lambda key: self.body_parts_xyc[key][1])][1]
            min_y = self.body_parts_xyc[min(self.body_parts_xyc, key=lambda key: self.body_parts_xyc[key][1])][1]

            return BBox(min_x, min_y, max_x, max_y)

        else:
            return BBox(0,0,0,0) #TODO proper invalid flag!

    def get_face_box(self, img_w, img_h, mode=0):
        """
        Get Face box compared to img size (w, h)
        :param img_w:
        :param img_h:
        :param mode:
        :return:
        """
        # SEE : https://github.com/ildoonet/tf-pose-estimation/blob/master/tf_pose/common.py#L13
        _NOSE = CocoPart.Nose.value
        _NECK = CocoPart.Neck.value
        _REye = CocoPart.REye.value
        _LEye = CocoPart.LEye.value
        _REar = CocoPart.REar.value
        _LEar = CocoPart.LEar.value

        _THRESHOLD_PART_CONFIDENCE = 0.2
        parts = [part for idx, part in self.body_parts.items() if part.score > _THRESHOLD_PART_CONFIDENCE]

        is_nose, part_nose = _include_part(parts, _NOSE)
        if not is_nose:
            return None

        size = 0
        is_neck, part_neck = _include_part(parts, _NECK)
        if is_neck:
            size = max(size, img_h * (part_neck.y - part_nose.y) * 0.8)

        is_reye, part_reye = _include_part(parts, _REye)
        is_leye, part_leye = _include_part(parts, _LEye)
        if is_reye and is_leye:
            size = max(size, img_w * (part_reye.x - part_leye.x) * 2.0)
            size = max(size,
                       img_w * math.sqrt((part_reye.x - part_leye.x) ** 2 + (part_reye.y - part_leye.y) ** 2) * 2.0)

        if mode == 1:
            if not is_reye and not is_leye:
                return None

        is_rear, part_rear = _include_part(parts, _REar)
        is_lear, part_lear = _include_part(parts, _LEar)
        if is_rear and is_lear:
            size = max(size, img_w * (part_rear.x - part_lear.x) * 1.6)

        if size <= 0:
            return None

        if not is_reye and is_leye:
            x = part_nose.x * img_w - (size // 3 * 2)
        elif is_reye and not is_leye:
            x = part_nose.x * img_w - (size // 3)
        else:  # is_reye and is_leye:
            x = part_nose.x * img_w - size // 2

        x2 = x + size
        if mode == 0:
            y = part_nose.y * img_h - size // 3
        else:
            y = part_nose.y * img_h - _round(size / 2 * 1.2)
        y2 = y + size

        # fit into the image frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if _round(x2 - x) == 0.0 or _round(y2 - y) == 0.0:
            return None
        if mode == 0:
            return {"x": _round((x + x2) / 2),
                    "y": _round((y + y2) / 2),
                    "w": _round(x2 - x),
                    "h": _round(y2 - y)}
        else:
            return {"x": _round(x),
                    "y": _round(y),
                    "w": _round(x2 - x),
                    "h": _round(y2 - y)}

    def get_upper_body_box(self, img_w, img_h):
        """
        Get Upper body box compared to img size (w, h)
        :param img_w:
        :param img_h:
        :return:
        """

        if not (img_w > 0 and img_h > 0):
            raise Exception("img size should be positive")

        _NOSE = CocoPart.Nose.value
        _NECK = CocoPart.Neck.value
        _RSHOULDER = CocoPart.RShoulder.value
        _LSHOULDER = CocoPart.LShoulder.value
        _THRESHOLD_PART_CONFIDENCE = 0.3
        parts = [part for idx, part in self.body_parts.items() if part.score > _THRESHOLD_PART_CONFIDENCE]
        part_coords = [(img_w * part.x, img_h * part.y) for part in parts if
                       part.part_idx in [0, 1, 2, 5, 8, 11, 14, 15, 16, 17]]

        if len(part_coords) < 5:
            return None

        # Initial Bounding Box
        x = min([part[0] for part in part_coords])
        y = min([part[1] for part in part_coords])
        x2 = max([part[0] for part in part_coords])
        y2 = max([part[1] for part in part_coords])

        # # ------ Adjust heuristically +
        # if face points are detcted, adjust y value

        is_nose, part_nose = _include_part(parts, _NOSE)
        is_neck, part_neck = _include_part(parts, _NECK)
        torso_height = 0
        if is_nose and is_neck:
            y -= (part_neck.y * img_h - y) * 0.8
            torso_height = max(0, (part_neck.y - part_nose.y) * img_h * 2.5)
        #
        # # by using shoulder position, adjust width
        is_rshoulder, part_rshoulder = _include_part(parts, _RSHOULDER)
        is_lshoulder, part_lshoulder = _include_part(parts, _LSHOULDER)
        if is_rshoulder and is_lshoulder:
            half_w = x2 - x
            dx = half_w * 0.15
            x -= dx
            x2 += dx
        elif is_neck:
            if is_lshoulder and not is_rshoulder:
                half_w = abs(part_lshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)
            elif not is_lshoulder and is_rshoulder:
                half_w = abs(part_rshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)

        # ------ Adjust heuristically -

        # fit into the image frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if _round(x2 - x) == 0.0 or _round(y2 - y) == 0.0:
            return None
        return {"x": _round((x + x2) / 2),
                "y": _round((y + y2) / 2),
                "w": _round(x2 - x),
                "h": _round(y2 - y)}

    def oks_dist(self, other, sigmas=None):
        if sigmas is None:
            sigmas = self.kpt_oks_sigmas # For Openpose Keypoint order

        num_joints = len(sigmas)
        kappas = (sigmas * 2) ** 2
        area = self.get_bbox().area() + np.spacing(1)
        #print('Bbox area: {}.'.format(area))

        e = []
        for i in range(num_joints):
            if i in self.body_parts and i in other.body_parts: # keypoints is present in both skeletons -> compute distance
                dx = self.body_parts[i].x - other.body_parts[i].x
                dy = self.body_parts[i].y - other.body_parts[i].y

                e.append((dx**2 + dy**2) / (2 * area * kappas[i]))

        if len(e) > 0:
            e = np.array(e)
            oks = np.sum(np.exp(-e)) / e.shape[0]
        else:
            oks = 0.0

        return oks

    def __str__(self):
        return ', '.join([str(x) for x in self.body_parts.values()]) + ', total_score=%.2f\n' % self.score

    def __repr__(self):
        return self.__str__()


class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()


class PoseEstimator:
    def __init__(self):
        pass

    @staticmethod
    def estimate_paf(peaks, heat_mat, paf_mat, nms_px = 1):
        pafprocess.process_paf(peaks, heat_mat, paf_mat, nms_px)

        humans = []
        for human_id in range(pafprocess.get_num_humans()):
            human = Human([])
            is_added = False

            for part_idx in range(18):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    continue

                is_added = True
                human.body_parts[part_idx] = BodyPart(
                    '%d-%d' % (human_id, part_idx), part_idx,
                    float(pafprocess.get_part_x(c_idx)) / heat_mat.shape[2],
                    float(pafprocess.get_part_y(c_idx)) / heat_mat.shape[1],
                    pafprocess.get_part_score(c_idx)
                )
                human.body_parts_xyc[part_idx] = (human.body_parts[part_idx].x, human.body_parts[part_idx].y, human.body_parts[part_idx].score)

            if is_added:
                score = pafprocess.get_score(human_id)
                human.score = score
                humans.append(human)

        return humans

    @staticmethod
    def estimate_paf_2(peaks, heat_mat, paf_mat, nms_px = 1):
        pafprocess.process_paf_2(peaks, heat_mat, paf_mat, nms_px)

        humans = []
        for human_id in range(pafprocess.get_num_humans()):
            human = Human([])
            is_added = False

            for part_idx in range(18):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    continue

                is_added = True
                human.body_parts[part_idx] = BodyPart(
                    '%d-%d' % (human_id, part_idx), part_idx,
                    float(pafprocess.get_part_x(c_idx)) / heat_mat.shape[1],
                    float(pafprocess.get_part_y(c_idx)) / heat_mat.shape[0],
                    pafprocess.get_part_score(c_idx)
                )
                human.body_parts_xyc[part_idx] = (human.body_parts[part_idx].x, human.body_parts[part_idx].y, human.body_parts[part_idx].score)

            if is_added:
                score = pafprocess.get_score(human_id)
                human.score = score
                humans.append(human)

        return humans
    
    @staticmethod
    def estimate_single(heat_mat, part_threshold = 0.10, crop_info = None, score_thresh = None, skel_feedback = None, heat_mat_orig = None):
        _POST_PROCESS = True
        if score_thresh is None:
            score_thresh = part_threshold
        
        num_joints = heat_mat.shape[2]
        width = heat_mat.shape[1]
        heatmaps_reshaped = heat_mat.reshape((-1, num_joints))
        idx = np.argmax(heatmaps_reshaped, 0)
        
        if heat_mat_orig is not None:
            heat_mat_orig = heat_mat_orig.reshape((-1, num_joints))

        if skel_feedback is not None:
            if len(skel_feedback.person.keypoints) != num_joints:
                print('Error using skeleton feedback! expected {} joints but got {}. Feedback will not be used.'.format(num_joints, len(skel_feedback.person.keypoints)))
                skel_feedback = None

        idx = idx.reshape((num_joints, 1))
        preds = np.tile(idx, (1, 2))

        preds[:, 0] = (preds[:, 0]) % width # x
        preds[:, 1] = (preds[:, 1]) // width # y

        coords = preds.astype(np.float32)
        #t_coords = time.monotonic() - start

        # post-processing
        if _POST_PROCESS:
            heatmap_height = heat_mat.shape[0]
            heatmap_width = heat_mat.shape[1]
            for p in range(coords.shape[0]):
                hm = heat_mat[:,:,p]
                px = int(math.floor(coords[p][0] + 0.5))
                py = int(math.floor(coords[p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array([hm[py][px+1] - hm[py][px-1],
                                    hm[py+1][px]-hm[py-1][px]])
                    coords[p] += np.sign(diff) * .25
        #t_pp = time.monotonic() - start

        if crop_info is not None:
            crop_w = crop_info[2] - crop_info[0]
            crop_h = crop_info[3] - crop_info[1]
        humans = []
        human = Human([])
        n_joints = 0
        n_joints_score = 0
        running_score = 0.0

        if skel_feedback is not None: # Will use weighted mean of feedback and current detection.
            for part_idx in range(num_joints):
                heatmap_score = float(heatmaps_reshaped[idx[part_idx, 0], part_idx])
                feedback_score = skel_feedback.person.keypoints[part_idx].score
                feedback_joint = skel_feedback.person.keypoints[part_idx].joint # rel. coordinates w.r.t heatmap size
                if feedback_joint.x == 0.0 and feedback_joint.y == 0.0: # don't use invalid joint
                    feedback_score = 0.0
                
                if heatmap_score <= part_threshold and feedback_score <= part_threshold :
                    continue
                
                if heatmap_score > score_thresh or feedback_score > score_thresh:
                    running_score += (heatmap_score + feedback_score) / 2. if feedback_score > 0.0 else heatmap_score
                    n_joints_score += 1
                
                if crop_info is None: # relative coordinates
                    human.body_parts[part_idx] = BodyPart(
                            '%d-%d' % (0, part_idx), part_idx,
                            (heatmap_score * coords[part_idx][0] / heat_mat.shape[1] + feedback_score * feedback_joint.x) / (heatmap_score + feedback_score),
                            (heatmap_score * coords[part_idx][1] / heat_mat.shape[0] + feedback_score * feedback_joint.y) / (heatmap_score + feedback_score),
                            (heatmap_score + feedback_score) / 2. if feedback_score > 0.0 else heatmap_score)
                else: # absolute coordinates -> No feedback used
                    human.body_parts[part_idx] = BodyPart(
                            '%d-%d' % (0, part_idx), part_idx,
                            coords[part_idx][0] / heat_mat.shape[1] * crop_w + crop_info[0],
                            coords[part_idx][1] / heat_mat.shape[0] * crop_h + crop_info[1],
                            float(heatmaps_reshaped[idx[part_idx, 0], part_idx]))

                human.body_parts_xyc[part_idx] = (human.body_parts[part_idx].x, human.body_parts[part_idx].y, human.body_parts[part_idx].score)
                n_joints += 1
        else:
            for part_idx in range(num_joints):
                if heatmaps_reshaped[idx[part_idx, 0], part_idx] <= part_threshold:
                    continue
                
                if heatmaps_reshaped[idx[part_idx, 0], part_idx] > score_thresh:
                    running_score += heatmaps_reshaped[idx[part_idx, 0], part_idx]
                    n_joints_score += 1
                
                if crop_info is None: # relative coordinates
                    human.body_parts[part_idx] = BodyPart(
                            '%d-%d' % (0, part_idx), part_idx,
                            coords[part_idx][0] / heat_mat.shape[1],
                            coords[part_idx][1] / heat_mat.shape[0],
                            float(heatmaps_reshaped[idx[part_idx, 0], part_idx]))
                else: # absolute coordinates
                    human.body_parts[part_idx] = BodyPart(
                            '%d-%d' % (0, part_idx), part_idx,
                            coords[part_idx][0] / heat_mat.shape[1] * crop_w + crop_info[0],
                            coords[part_idx][1] / heat_mat.shape[0] * crop_h + crop_info[1],
                            float(heatmaps_reshaped[idx[part_idx, 0], part_idx]))

                if heat_mat_orig is not None:
                    human.body_parts_xyc[part_idx] = (human.body_parts[part_idx].x, human.body_parts[part_idx].y, human.body_parts[part_idx].score, float(heat_mat_orig[idx[part_idx, 0], part_idx])) # original heatmaps score should be used as depth-confidence.
                else:
                    human.body_parts_xyc[part_idx] = (human.body_parts[part_idx].x, human.body_parts[part_idx].y, human.body_parts[part_idx].score)
                n_joints += 1
            
        if n_joints_score > 0:
            human.score = running_score / n_joints_score # score is initialized as zero

        if n_joints > 0:
            humans.append(human)
        
        #t_end = time.monotonic() - start
        #print('\n\testimator times: argmax: {} ms; coords: {} ms; pp: {} ms; end: {} ms.'.format(t_argmax * 1000, t_coords * 1000, t_pp * 1000, t_end * 1000))
        return humans

    @staticmethod
    def humans_to_json(humans, roi = None, crop_info = None):
        jsondata = {}
        jsondata['timestamp'] = TimestampMillisec64()
        jsondata['people'] = []
        
        if roi is not None:
            jsondata['crop_info'] = [roi.x_offset, roi.y_offset, roi.width, roi.height] # center, w, h !! Achtung: x- and y- offset actually are x- and y- center coordinates !!
        elif crop_info is not None:
            jsondata['crop_info'] = crop_info # x0, y0, x1, y1

        for human in humans:
            person = {}
            person['keypoints'] = human.body_parts_xyc
            person['score'] = human.score
            person['id'] = human.id
            person['bbox'] = human.get_bbox()
            
            jsondata['people'].append(person)

        return json.dumps(jsondata)

    @staticmethod
    def draw_humans(pil_img, humans, imgcopy=False, keep_aspect_ratio=False, simple_baselines=False, h36m=False):
        _CONF_THRESHOLD_DRAW = 0.0
        if imgcopy:
            pil_img = pil_img.copy()
        image_w, image_h = pil_img.size[:2]
        if keep_aspect_ratio:
            imsize = max(image_h, image_w) # assuming square network input -> coordinates are relative to padded, square image
            w_offset = (imsize - image_w) // 2
            h_offset = (imsize - image_h) // 2
        else:
            w_offset = 0
            h_offset = 0
        draw = ImageDraw.Draw(pil_img)
        num_joints = 18 if not simple_baselines else 17
        
        if simple_baselines:
            if h36m:
                pairs = common.H36MPairs
            else:
                pairs = common.CocoPairsSB
        else:
            pairs = common.CocoPairsRender
            
        for human in humans:
            centers = {}
            body_parts = {}
            # draw point
            for i in range(num_joints):
                if i not in human.body_parts.keys() or human.body_parts[i].score < _CONF_THRESHOLD_DRAW:
                    continue

                body_part = human.body_parts[i]
                if not keep_aspect_ratio:
                    center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                else:
                    center = (int(body_part.x * imsize - w_offset + 0.5), int(body_part.y * imsize - h_offset + 0.5))

                centers[i] = center
                body_parts[i] = body_part
                draw.ellipse([(center[0]-2, center[1]-2), (center[0]+2, center[1]+2)], fill=common.CocoColors[i])
                #cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)

            # draw line
            for pair_order, pair in enumerate(pairs):
                if pair[0] not in centers.keys() or pair[1] not in centers.keys():
                    continue

                draw.line([centers[pair[0]], centers[pair[1]]], fill=common.CocoColors[pair_order], width=3)
                # npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
                #cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)

            # draw bounding box
            x1, y1, x2, y2 = human.get_bbox()
            if keep_aspect_ratio:
                x1 = int(x1 * imsize - w_offset + 0.5)-6
                y1 = int(y1 * imsize - h_offset + 0.5)-6
                x2 = int(x2 * imsize - w_offset + 0.5)+6
                y2 = int(y2 * imsize - h_offset + 0.5)+6
            else:
                x1 = int(x1 * image_w + 0.5)-6
                y1 = int(y1 * image_h + 0.5)-6
                x2 = int(x2 * image_w + 0.5)+6
                y2 = int(y2 * image_h + 0.5)+6
            draw.rectangle([(x1, y1), (x2, y2)], outline = common.CocoColors[human.id % len(common.CocoColors)], fill = None, width = 3)
            draw.text((x1, y1 - 24), 'ID: {}'.format(human.id), fill=common.CocoColors[human.id % len(common.CocoColors)], font = ImageFont.truetype('Exo2-SemiBold.ttf', 22))

            for i in body_parts:
                if simple_baselines:
                    if i == 0: # Nose -> print to bottom of joint
                        draw.text((centers[i][0], centers[i][1] + 6), '{:.2f}'.format(body_parts[i].score), fill=common.CocoColors[i], font = ImageFont.truetype('Exo2-SemiBold.ttf', 20))
                    elif i % 2 != 0: # Impair : left body half
                        draw.text((centers[i][0] + 6, centers[i][1]), '{:.2f}'.format(body_parts[i].score), fill=common.CocoColors[i], font = ImageFont.truetype('Exo2-SemiBold.ttf', 20))
                    else: # Pari: right body half
                        draw.text((centers[i][0] - 42, centers[i][1]), '{:.2f}'.format(body_parts[i].score), fill=common.CocoColors[i], font = ImageFont.truetype('Exo2-SemiBold.ttf', 20))
                else:
                    if i <=1 or (i >=5 and i <=7):
                        draw.text((centers[i][0] + 6, centers[i][1]), '{:.2f}'.format(body_parts[i].score), fill=common.CocoColors[i], font = ImageFont.truetype('Exo2-SemiBold.ttf', 20))
                    elif i <=4:
                        draw.text((centers[i][0] - 42, centers[i][1]), '{:.2f}'.format(body_parts[i].score), fill=common.CocoColors[i], font = ImageFont.truetype('Exo2-SemiBold.ttf', 20))
        return pil_img

    @staticmethod
    def draw_humans_abs(pil_img, humans, thresh_draw = 0.20):
        draw = ImageDraw.Draw(pil_img)
        num_joints = 17
        pairs = common.CocoPairsSB
        for human in humans:
            centers = {}
            body_parts = {}

            # draw point
            for i in range(num_joints):
                if i not in human.body_parts.keys() or human.body_parts[i].score < thresh_draw:
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x + 0.5), int(body_part.y + 0.5))

                centers[i] = center
                body_parts[i] = body_part
                draw.ellipse([(center[0]-2, center[1]-2), (center[0]+2, center[1]+2)], fill=common.CocoColors[i])

            # draw line
            for pair_order, pair in enumerate(pairs):
                if pair[0] not in centers.keys() or pair[1] not in centers.keys():
                    continue

                draw.line([centers[pair[0]], centers[pair[1]]], fill=common.CocoColors[pair_order], width=3)

            # draw bounding box
            x1, y1, x2, y2 = human.get_bbox()
            x1 = int(x1 + 0.5)-6
            y1 = int(y1 + 0.5)-6
            x2 = int(x2 + 0.5)+6
            y2 = int(y2 + 0.5)+6

            draw.rectangle([(x1, y1), (x2, y2)], outline = common.CocoColors[human.id % len(common.CocoColors)], fill = None, width = 3)
            draw.text((x1, y1 - 24), 'ID: {}, conf: {:.2f}'.format(human.id, human.score), fill=common.CocoColors[human.id % len(common.CocoColors)], font = ImageFont.truetype('Exo2-SemiBold.ttf', 22))

        return pil_img

    @staticmethod
    def draw_humans_svg(doc, humans, imsize):
        from edgetpuvision import svg
        _CONF_THRESHOLD_DRAW = 0.0
        
        image_h = imsize[0]
        image_w = imsize[1]
        
        for human in humans:
            centers = {}
            # draw point
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys() or human.body_parts[i].score < _CONF_THRESHOLD_DRAW:
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_h + 0.5), int(body_part.y * image_w + 0.5))
                centers[i] = center
                doc += svg.Circle(cx=center[0], cy=center[1], r=3,
                        style='stroke:%s;fill:%s' % (svg.rgb(common.CocoColors[i]), svg.rgb(common.CocoColors[i])), _class='body')
                #draw.ellipse([(center[0]-2, center[1]-2), (center[0]+2, center[1]+2)], fill=common.CocoColors[i])

            # draw line
            for pair_order, pair in enumerate(common.CocoPairsRender):
                if pair[0] not in centers.keys() or pair[1] not in centers.keys():
                    continue

                doc += svg.Line(x1=centers[pair[0]][0], y1=centers[pair[0]][1], x2=centers[pair[1]][0], y2=centers[pair[1]][1],
                        style='stroke:%s' % svg.rgb(common.CocoColors[pair_order]), _class='body')
                #draw.line([centers[pair[0]], centers[pair[1]]], fill=common.CocoColors[pair_order], width=3)
        return doc



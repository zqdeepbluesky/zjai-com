# -*- coding: utf-8 -*-
# @Time    : 7/13/2018 1:52 PM
# @Author  : sunyonghai
# @File    : predict.py
# @Software: ZJ_AI

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from utils.blob import im_list_to_blob
from utils.timer import Timer
from model.config import cfg
from model.bbox_transform import bbox_transform_inv
from model.nms_wrapper import nms



def _detect_image(sess, net, im):
    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
    _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])
    boxes = rois[:, 1:5] / im_scales[0]
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))
    return scores, pred_boxes

def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes

def _get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    return blobs, im_scale_factors

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    processed_ims = []
    im_scale_factors = []
    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)
    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)

def _get_thresh_label(class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    boxes = np.zeros((1,5),dtype=np.float32)
    cls_list = []
    score_list=[]
    flag = 1
    if len(inds) == 0:
        flag=0
        return boxes,cls_list,flag,score_list
    count=0
    for i in inds:
        bbox = list(map(int,dets[i, :4]))
        score = dets[i, -1]
        bbox.append(score)
        bbox = np.array(bbox)
        if count ==0:
            boxes[0,:]= bbox
        else:
            boxes=np.row_stack((boxes,bbox))
        count+=1
        cls_list.append(class_name)
        score_list.append(score)
    return boxes,cls_list,flag,score_list

def cal_time(func):
    def _cal_time(*args, **kwargs):
        timer=Timer()
        timer.tic()
        result_data=func(*args, **kwargs)
        timer.toc()
        print("deal this func have need {}s".format(timer.average_time))
        print("this images have predict {} object: {}".format(len(result_data),result_data))
        return result_data
    return _cal_time

@cal_time
def predict_image(sess,net,im,CLASSES):

    # Detect all object classes and regress object bounds
    scores, boxes = _detect_image(sess, net, im)
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    result_data=[]
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        box, classname, flag, score = _get_thresh_label(cls, dets, thresh=CONF_THRESH)
        if flag == -1:
            continue
        for i in range(len(classname)):
            result_data.append("{},{:.3f},{},{},{},{}".format(classname[i],score[i],int(box[i, 0]),int(box[i, 1]),int(box[i, 2]),int(box[i, 3])))
    return result_data

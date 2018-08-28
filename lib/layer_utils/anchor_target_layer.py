# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 8/9/2018 4:34 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from model.config import cfg
import numpy as np
import numpy.random as npr
from utils.cython_bbox import bbox_overlaps
from model.bbox_transform import bbox_transform

def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
    #剔除越出边界的roi,计算边界偏移值,选出前后景,初始化权重
    """Same as the anchor target layer in original Fast/er RCNN """
    A = num_anchors  #9
    total_anchors = all_anchors.shape[0] #候选框量,w*h*9/256
    K = total_anchors / num_anchors  #特征点

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3] #[h,w]

    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0] #未超过边界的下标

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]  #提取未超过边界的候选框

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)  #初始化label,并补为-1,0为背景,1为前景,-1忽略
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps( np.ascontiguousarray(anchors, dtype=np.float), np.ascontiguousarray(gt_boxes, dtype=np.float))
    #计算候选框与真实框的重合度 --重叠面积/（roi面积+GT面积-重叠面积），[w*h*9/256,len(ge_boxes)]
    argmax_overlaps = overlaps.argmax(axis=1)  #每行与真实框重合度最大的候选框下标
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps] #提取 重合度最大的候选框 [9*A,1]
    gt_argmax_overlaps = overlaps.argmax(axis=0)  #每列与真实框重合度最大的候选框下标
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]  #从全部rois 中提取重合度最高的 [1,len(gt_boxes)]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]  #

    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:  #先按照rpn的阈值挑选bg
        # assign bg labels first so that positive labels can clobber them
        # first set the negatives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0  #重合度少于0.3的视为背景

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1  #与每个gt_box重合度最高的roi视为前景

    # fg label: above threshold IOU
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1  #重合度大于0.7的视为前景

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:  #后按照rpn的阈值挑选bg
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)  #rpn前后景比例,提取一定批次数量的roi
    fg_inds = np.where(labels == 1)[0]   #找到前景下标
    if len(fg_inds) > num_fg:   #随机提取 一半batchsize个前景，正样本，其余忽略
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)  #剩余的背景数
    bg_inds = np.where(labels == 0)[0]   #找到背景下标
    if len(bg_inds) > num_bg:   # 随机提取 一半batchsize个背景，负样本，其余忽略
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
    #前景，背景label数目一致
    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32) #初始化label边界
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])   #计算得到目标偏移值,[len(gt_boxes),4]

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)  #初始化
    # only the positive ones have regression targets
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS) #设置前景权重 (1,1,1,1)

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32) #初始化
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:  #正例权重为负
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)  #正例样本总数
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples #初始化正例样本权重,1/正样本数
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples #
    else: #为正,(0,1)范围内
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) & (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT / np.sum(labels == 1)) #初始化权重 x/正样本数
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) / np.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights #权重赋值
    bbox_outside_weights[labels == 0, :] = negative_weights

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    # labels
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets.reshape((1, height, width, A * 4))

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4))

    rpn_bbox_outside_weights = bbox_outside_weights
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):  #重整data
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

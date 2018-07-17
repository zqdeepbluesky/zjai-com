# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)  #随机选择区间内的整数
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0),'num_images ({}) must divide BATCH_SIZE ({})'.format(num_images, cfg.TRAIN.BATCH_SIZE)    #batchSize必须能够整除每次输入参数

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)   #缩放后的图像，并装在blob，缩放系数

    blobs = {'data': im_blob}  #data存放图像

    assert len(im_scales) == 1, "Single batch only"  #???
    assert len(roidb) == 1, "Single batch only"      #???

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
        # Include all ground truth boxes
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)   #生成随机矩阵，gt_inds*5
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]  #修改boxes尺寸
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]   #获取真实的类别序号
    blobs['gt_boxes'] = gt_boxes         #为甚都是blob[0]???
    blobs['im_info'] = np.array(
        [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
        dtype=np.float32)

    return blobs

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)  #每次输入图像数
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])  #读取图像
        if roidb[i]['flipped']:     #被翻转
            im = im[:, ::-1, :]       #图像翻转
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]  #设置size
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)   #得到缩放后的图像和缩放系数
        im_scales.append(im_scale)   #存放起缩放系数
        processed_ims.append(im)     #存放缩放后的图像

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)   #利用blob最大的框架来装在缩放后的图像

    return blob, im_scales

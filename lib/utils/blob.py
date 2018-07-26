# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)  #长宽的最大值
    num_images = len(ims)     #获取每次输入的图像参数
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
    for i in range(num_images):
        im = ims[i]  #获取图像
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im   #填充图像，其余为0

    return blob


def prep_im_for_blob(im, pixel_means, target_size, max_size):  #max_size=1000,target_size=600
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape   #获得图像形状
    im_size_min = np.min(im_shape[0:2])   #size最小值，长宽
    im_size_max = np.max(im_shape[0:2])   #size最大值，长宽
    im_scale = float(target_size) / float(im_size_min)  #比例值
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:   #计算缩放后最大size值，如果大于max_size，那缩放系数为最大/最大
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)   #根据缩放系数对图像进行缩放

    return im, im_scale

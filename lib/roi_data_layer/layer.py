# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import time

class RoIDataLayer(object):
  """Fast R-CNN data layer used for training."""

  def __init__(self, roidb, num_classes, random=False):
    """Set the roidb to be used by this layer during training."""
    self._roidb = roidb
    self._num_classes = num_classes
    # Also set a random flag
    self._random = random
    self._shuffle_roidb_inds()

  def _shuffle_roidb_inds(self):
    """Randomly permute the training roidb."""
    # If the random flag is set, 
    # then the database is shuffled according to system time
    # Useful for the validation set
    if self._random:        #随机数参数,trainval为false，test为true
      st0 = np.random.get_state()
      millis = int(round(time.time() * 1000)) % 4294967295
      np.random.seed(millis)
    
    if cfg.TRAIN.ASPECT_GROUPING:     #False
      widths = np.array([r['width'] for r in self._roidb])  #得到长度
      heights = np.array([r['height'] for r in self._roidb]) #得到宽度
      horz = (widths >= heights)    #得到长宽比较值，长大于宽为1，否则为0
      vert = np.logical_not(horz)   #取反，长大于宽为0，否则为1
      horz_inds = np.where(horz)[0] #找到长大于宽的图像下标
      vert_inds = np.where(vert)[0] #找到宽大于长的图像下标
      inds = np.hstack((            #合并，成二维
          np.random.permutation(horz_inds),
          np.random.permutation(vert_inds)))
      inds = np.reshape(inds, (-1, 2))
      row_perm = np.random.permutation(np.arange(inds.shape[0]))  #随机打乱顺序
      inds = np.reshape(inds[row_perm, :], (-1,))
      self._perm = inds
    else:
      self._perm = np.random.permutation(np.arange(len(self._roidb)))  #随机打乱下标
    # Restore the random state
    if self._random:
      np.random.set_state(st0)
      
    self._cur = 0

  def _get_next_minibatch_inds(self):
    """Return the roidb indices for the next minibatch."""
    if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):  #roidb小于输入图像个数
      self._shuffle_roidb_inds()

    db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
    #print(self._perm)     #打乱之后的roidb下标
    self._cur += cfg.TRAIN.IMS_PER_BATCH

    return db_inds

  def _get_next_minibatch(self):
    """Return the blobs to be used for the next minibatch.

    If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
    separate process and made available through self._blob_queue.
    """
    db_inds = self._get_next_minibatch_inds()  #得到下一个最小batch的下标,若参数为1则len为1，为2则len为2
    minibatch_db = [self._roidb[i] for i in db_inds] #取下标对应的roidb
    return get_minibatch(minibatch_db, self._num_classes)
      
  def forward(self):
    """Get blobs and copy them into this layer's top blob vector."""
    blobs = self._get_next_minibatch()
    return blobs

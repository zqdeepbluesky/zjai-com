# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import PIL
from math import *
from utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
from model.config import cfg
from lib.datasets import data_augment


class imdb(object):
    """Image database."""

    def __init__(self, name, classes=None):
        self._name = name
        if not classes:
            self._classes = []
        else:
            self._classes = classes
        self._num_classes = len(self._classes)
        self._image_index = []
        self._obj_proposer = 'gt'
        self._roidb = None
        self._roidb_handler = self.default_roidb
        # Use this dict for storing dataset specific config options
        self.config = {}

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')
        self.roidb_handler = method

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def num_images(self):
        return len(self.image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def _get_widths(self):
        return [PIL.Image.open(self.image_path_at(i)).size[0] for i in range(self.num_images)]

    def append_hor_flipped_images(self):  #水平翻转
        num_images = self.num_images
        sizes =[PIL.Image.open(self.image_path_at(i)).size for i in range(self.num_images)]
        print("finish get_widths")
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()   #如果roidb还未加载时，会先加载，并保存,图片的object
            oldx1 = boxes[:, 0].copy()  #xmin
            oldx2 = boxes[:, 2].copy()  #xmax
            boxes[:, 0] = sizes[i][0] - oldx2
            boxes[:, 2] = sizes[i][0] - oldx1
            index=np.where(boxes[:, 2]< boxes[:, 0])
            boxes[index,0]=0
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes': boxes,
                     'gt_overlaps': self.roidb[i]['gt_overlaps'],
                     'gt_classes': self.roidb[i]['gt_classes'],
                     'hor_flipped': True,
                     'width':sizes[i][0],
                     'height':sizes[i][1]}

            for key in self.roidb[i].keys():
                if key not in entry.keys():
                    entry[key] = self.roidb[i][key]
            self.roidb.append(entry)
        self._image_index = self._image_index *2

    def append_ver_flipped_images(self):  #竖直翻转
        num_images = self.num_images
        sizes = [PIL.Image.open(self.image_path_at(i)).size for i in range(self.num_images)]
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()   #如果roidb还未加载时，会先加载，并保存,图片的object
            oldx1 = boxes[:, 1].copy()  #ymin
            oldx2 = boxes[:, 3].copy()  #ymax
            boxes[:, 1] = sizes[i][1] - oldx2
            boxes[:, 3] = sizes[i][1] - oldx1
            index = np.where(boxes[:, 3] < boxes[:, 1])
            boxes[index, 1] = 0
            assert (boxes[:, 3] >= boxes[:, 1]).all()
            entry = {'boxes': boxes,
                     'gt_overlaps': self.roidb[i]['gt_overlaps'],
                     'gt_classes': self.roidb[i]['gt_classes'],
                     'ver_flipped': True,
                     'width': sizes[i][0],
                     'height': sizes[i][1]
                     }

            for key in self.roidb[i].keys():
                if key not in entry.keys():
                    entry[key] = self.roidb[i][key]
            self.roidb.append(entry)
        self._image_index = self._image_index * 2

    def append_bright_adjuest_images(self):
        num_images = self.num_images
        sizes = [PIL.Image.open(self.image_path_at(i)).size for i in range(self.num_images)]
        error_num=0
        for gamma in cfg.TRAIN.BRIGHT_ADJUEST_SCALE:
            if gamma==1 or gamma<=0:
                error_num+=1
                continue
            for i in range(num_images):
                boxes = self.roidb[i]['boxes'].copy()
                entry = {'boxes': boxes,
                         'gt_overlaps': self.roidb[i]['gt_overlaps'],
                         'gt_classes': self.roidb[i]['gt_classes'],
                         'bright_scale': gamma,
                         'width': sizes[i][0],
                         'height': sizes[i][1]
                         }

                for key in self.roidb[i].keys():
                    if key not in entry.keys():
                        entry[key] = self.roidb[i][key]
                self.roidb.append(entry)
        self._image_index += self._image_index * (len(cfg.TRAIN.BRIGHT_ADJUEST_SCALE)-error_num)

    def append_rotate_adjuest_images(self):
        num_images = self.num_images
        error_num=0
        sizes = [PIL.Image.open(self.image_path_at(i)).size for i in range(self.num_images)]
        for angle in cfg.TRAIN.ROTATE_ADJUEST_ANGLE:
            if angle==0 or angle==360:
                error_num+=1
                continue
            for i in range(num_images):
                boxes = self.roidb[i]['boxes'].copy()
                size=[sizes[i][0],sizes[i][1]]
                boxes,widthNew,heightNew=data_augment._rotate_boxes(boxes, size, angle)
                entry = {'boxes': boxes,
                         'gt_overlaps': self.roidb[i]['gt_overlaps'],
                         'gt_classes': self.roidb[i]['gt_classes'],
                         'rotate_angle':angle,
                         'width': int(widthNew),
                         'height': int(heightNew)}

                for key in self.roidb[i].keys():
                    if key not in entry.keys():
                        entry[key] = self.roidb[i][key]
                self.roidb.append(entry)
        self._image_index += self._image_index * (len(cfg.TRAIN.ROTATE_ADJUEST_ANGLE)-error_num)

    def append_shift_adjuest_images(self):
        num_images = self.num_images
        this_image_index=self._image_index
        sizes = [PIL.Image.open(self.image_path_at(i)).size for i in range(self.num_images)]
        offset=(cfg.TRAIN.SHIFT_ADJUEST_X,cfg.TRAIN.SHIFT_ADJUEST_Y)
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            size = [sizes[i][0], sizes[i][1]]
            boxes = data_augment._shift_boxes(boxes, size, offset)
            if len(boxes)==0:
                this_image_index.remove(this_image_index[i])
                continue

            entry = {'boxes': boxes,
                     'gt_overlaps': self.roidb[i]['gt_overlaps'],
                     'gt_classes': self.roidb[i]['gt_classes'],
                     'shift_x':offset[0],
                     'shift_y':offset[1],
                     'width': size[0],
                     'height': size[1]}

            for key in self.roidb[i].keys():
                if key not in entry.keys():
                    entry[key] = self.roidb[i][key]
            self.roidb.append(entry)
        self._image_index += this_image_index

    def append_zoom_adjuest_images(self):
        num_images = self.num_images
        sizes = [PIL.Image.open(self.image_path_at(i)).size for i in range(self.num_images)]
        scales=cfg.TRAIN.ZOOM_ADJUEST_SCALE
        error_num=0
        for scale in scales:
            if scale[0]==1 and scale[1]==1:
                error_num+=1
                continue
            for i in range(num_images):
                boxes = self.roidb[i]['boxes'].copy()
                size = [sizes[i][0], sizes[i][1]]
                boxes = data_augment._zoom_boxes(boxes, scale)

                entry = {'boxes': boxes,
                         'gt_overlaps': self.roidb[i]['gt_overlaps'],
                         'gt_classes': self.roidb[i]['gt_classes'],
                         'zoom_x':scale[0],
                         'zoom_y':scale[1],
                         'width': int(size[0]*scale[0]),
                         'height': int(size[1]*scale[1])}

                for key in self.roidb[i].keys():
                    if key not in entry.keys():
                        entry[key] = self.roidb[i][key]
                self.roidb.append(entry)
        self._image_index += self._image_index * (len(cfg.TRAIN.ZOOM_ADJUEST_SCALE) - error_num)

    def append_random_crop_images(self):
        num_images=self.num_images
        sizes = [PIL.Image.open(self.image_path_at(i)).size for i in range(self.num_images)]
        crop_sizes=cfg.TRAIN.CROP_SIZE
        scale=cfg.TRAIN.RESIZE_SCALE
        this_image_index=[]
        for crop_size in crop_sizes:
            for i in range(num_images):
                boxes = self.roidb[i]['boxes'].copy()
                resize_scale = data_augment.cal_scale(sizes[i], scale)
                img_size = (int(sizes[i][0] / resize_scale), int(sizes[i][1] / resize_scale))
                boxes = data_augment.resize_box(boxes, resize_scale)
                print(img_size,crop_size,resize_scale)
                if img_size[0] >= crop_size[0] and img_size[1] >= crop_size[1] and resize_scale>=1:
                    position_list = ['lu', 'ld', 'ru', 'rd', 'm']
                    crop_bboxs = data_augment.create_crop_bbox(img_size, crop_size)
                    for j in range(len(crop_bboxs)):
                        new_boxes = data_augment.cal_random_crop_box(boxes, crop_bboxs[j])
                        if len(new_boxes)==0:
                            continue
                        entry = {'boxes': new_boxes,
                                 'gt_overlaps': self.roidb[i]['gt_overlaps'],
                                 'gt_classes': self.roidb[i]['gt_classes'],
                                 'position':position_list[j],
                                 'crop_size_width':crop_size[0],
                                 'crop_size_height':crop_size[1],
                                 'width': crop_bboxs[j].shape[1],
                                 'height': crop_bboxs[j].shape[0]}
                        for key in self.roidb[i].keys():
                            if key not in entry.keys():
                                entry[key] = self.roidb[i][key]
                        self.roidb.append(entry)
                        this_image_index.append(self._image_index[i])
        self._image_index+=this_image_index




    def evaluate_recall(self, candidate_boxes=None, thresholds=None, area='all', limit=None):
        """Evaluate detection proposal recall metrics.

        Returns:
            results: dictionary of results with keys
                'ar': average recall
                'recalls': vector recalls at each IoU overlap threshold
                'thresholds': vector of IoU overlap thresholds
                'gt_overlaps': vector of all ground-truth overlaps
        """
        # Record max overlap value for each gt box
        # Return vector of overlap values
        areas = {'all': 0, 'small': 1, 'medium': 2, 'large': 3, '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
        area_ranges = [[0 ** 2, 1e5 ** 2],  # all
                       [0 ** 2, 32 ** 2],  # small
                       [32 ** 2, 96 ** 2],  # medium
                       [96 ** 2, 1e5 ** 2],  # large
                       [96 ** 2, 128 ** 2],  # 96-128
                       [128 ** 2, 256 ** 2],  # 128-256
                       [256 ** 2, 512 ** 2],  # 256-512
                       [512 ** 2, 1e5 ** 2],  # 512-inf
                       ]
        assert area in areas, 'unknown area range: {}'.format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = np.zeros(0)
        num_pos = 0
        for i in range(self.num_images):
            # Checking for max_overlaps == 1 avoids including crowd annotations
            # (...pretty hacking :/)
            max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(axis=1)
            gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) & (max_gt_overlaps == 1))[0]
            gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
            gt_areas = self.roidb[i]['seg_areas'][gt_inds]
            valid_gt_inds = np.where((gt_areas >= area_range[0]) & (gt_areas <= area_range[1]))[0]
            gt_boxes = gt_boxes[valid_gt_inds, :]
            num_pos += len(valid_gt_inds)

            if candidate_boxes is None:
                # If candidate_boxes is not supplied, the default is to use the
                # non-ground-truth boxes from this roidb
                non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
                boxes = self.roidb[i]['boxes'][non_gt_inds, :]
            else:
                boxes = candidate_boxes[i]
            if boxes.shape[0] == 0:
                continue
            if limit is not None and boxes.shape[0] > limit:
                boxes = boxes[:limit, :]

            overlaps = bbox_overlaps(boxes.astype(np.float),gt_boxes.astype(np.float))

            _gt_overlaps = np.zeros((gt_boxes.shape[0]))
            for j in range(gt_boxes.shape[0]):
                # find which proposal box maximally covers each gt box
                argmax_overlaps = overlaps.argmax(axis=0)
                # and get the iou amount of coverage for each gt box
                max_overlaps = overlaps.max(axis=0)
                # find which gt box is 'best' covered (i.e. 'best' = most iou)
                gt_ind = max_overlaps.argmax()
                gt_ovr = max_overlaps.max()
                assert (gt_ovr >= 0)
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the iou coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert (_gt_overlaps[j] == gt_ovr)
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1
            # append recorded iou coverage level
            gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

        gt_overlaps = np.sort(gt_overlaps)
        if thresholds is None:
            step = 0.05
            thresholds = np.arange(0.5, 0.95 + 1e-5, step)
        recalls = np.zeros_like(thresholds)
        # compute recall for each iou threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        # ar = 2 * np.trapz(recalls, thresholds)
        ar = recalls.mean()
        return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds, 'gt_overlaps': gt_overlaps}

    def create_roidb_from_box_list(self, box_list, gt_roidb):
        assert len(box_list) == self.num_images,'Number of boxes must match number of ground-truth images'
        roidb = []
        for i in range(self.num_images):
            boxes = box_list[i]
            num_boxes = boxes.shape[0]
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

            if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
                gt_boxes = gt_roidb[i]['boxes']
                gt_classes = gt_roidb[i]['gt_classes']
                gt_overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))
                argmaxes = gt_overlaps.argmax(axis=1)
                maxes = gt_overlaps.max(axis=1)
                I = np.where(maxes > 0)[0]
                overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

            overlaps = scipy.sparse.csr_matrix(overlaps)
            roidb.append({
                'boxes': boxes,
                'gt_classes': np.zeros((num_boxes,), dtype=np.int32),
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': np.zeros((num_boxes,), dtype=np.float32),
            })
        return roidb

    @staticmethod
    def merge_roidbs(a, b):
        assert len(a) == len(b)
        for i in range(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'], b[i]['gt_classes']))
            a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'], b[i]['gt_overlaps']])
            a[i]['seg_areas'] = np.hstack((a[i]['seg_areas'], b[i]['seg_areas']))
        return a

    def competition_mode(self, on):
        """Turn competition mode on or off."""
        pass

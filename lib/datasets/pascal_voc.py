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
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from model.config import cfg
from lib.extra_utils import io_utils

DEBUG = True
def read_classes(path):
    classes = []
    with open(path, "r") as f:
        lineList = f.readlines()
        for line in lineList:
            line = line.replace("\n", "")
            classes.append(line)
    return tuple(classes)

def load_file_dict(path):
    with open(path, 'rb') as fid:
        try:
            file_dict = pickle.load(fid)
        except:
            file_dict = pickle.load(fid, encoding='bytes')
    return file_dict

class pascal_voc(imdb):
    def __init__(self, image_set, year,package_name):
        name = 'voc_' + year + '_' + image_set
        self._year = year
        self._image_set = image_set
        self.package_name=package_name
        if DEBUG:
            # self._devkit_path = os.path.abspath(os.path.join(self._get_default_path(), "train_data", 'all_train_data2'))
            self._devkit_path = os.path.abspath(os.path.join(self._get_default_path(), "train_data"))
            self._classes = read_classes(os.path.join(cfg.ROOT_DIR,'experiments', 'classes_cfgs', cfg.CLASSES_FILE))
        else:
            self._devkit_path = os.path.abspath(os.path.join(self._get_default_path(), "train_data", 'VOC2007_origin'))
            self._classes = read_classes(os.path.join(cfg.ROOT_DIR,'experiments', 'classes_cfgs', 'voc_classes.txt'))

        imdb.__init__(self, name, self._classes)
        # self._devkit_path = self._get_default_path()   #返回基础路径
        self._data_path = [os.path.join(self._devkit_path,file_name) for file_name in self.package_name]
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))  #弄成序号
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb    #返回基础的roidb
        # self._file_dict = self._load_file_dict()
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'rpn_file': None}

        assert os.path.exists(self._devkit_path),'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        for i in range(len(self._data_path)):
            assert os.path.exists(self._data_path[i]),'Path does not exist: {}'.format(self._data_path[i])

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        file_dict_dir=os.path.join(cfg.ROOT_DIR,'data','file_dict')
        file_list = load_file_dict(os.path.join(file_dict_dir, "{}_file_dict.txt".format("+".join(self.package_name))))
        image_path = os.path.join(file_list[index], 'JPEGImages', index + self._image_ext)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def _wrote_file_dict(self,image_index,image_file_path):
        file_dict={}
        file_dict_dir = os.path.join(cfg.ROOT_DIR, 'data', 'file_dict')
        if not os.path.exists(file_dict_dir):
            io_utils.mkdir(file_dict_dir)
        for i in range(len(image_index)):
            file_dict[image_index[i]]=image_file_path[i]
        with open(os.path.join(file_dict_dir,'{}_file_dict.txt'.format("+".join(self.package_name))), 'wb') as f:
            pickle.dump(file_dict, f, pickle.HIGHEST_PROTOCOL)

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/Main/val.txt
        image_index=[]
        image_file_path=[]
        for i in range(len(self._data_path)):
            image_set_file = os.path.join(self._data_path[i], 'ImageSets', 'Main', self._image_set + '.txt')
            assert os.path.exists(image_set_file),'Path does not exist: {}'.format(image_set_file)
            with open(image_set_file) as f:
                for x in f.readlines():
                    image_index.append(x.strip())
                    image_file_path.append(os.path.join(self._data_path[i]))
        self._wrote_file_dict(image_index,image_file_path)
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        读取并返回图片gt的db。这个函数就是将图片的gt加载进来。
        其中，pascal_voc图片的gt信息在XML文件中（这个XML文件是pascal_voc数据集本身提供的）
        并且，图片的gt被提前放在了一个.pkl文件里面。（这个.pkl文件需要我们自己生成，代码就在该函数中）

        This function loads/saves from/to a cache file to speed up future calls.
        之所以会将图片的gt提前放在一个.pkl文件里面，是为了不用每次都再重新读图片的gt，直接加载这个文件就可以了，可以提升速度。

        注意：如果你再次训练的时候修改了train数据库，增加或者删除了一些数据，再想重新训练的时候，一定要先删除这个.pkl文件！！！！！！因为如果不删除的话，就会自动加载旧的pkl文件，而不会生成新的pkl文件。一定别忘了！
        """
        cache_file = os.path.join(self.cache_path,'{}_{}_gt_roidb.pkl'.format("+".join(self.package_name),self.name))
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index) for index in self._image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename),'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        从XML文件中获取图片信息和gt
        """
        file_dict_dir = os.path.join(cfg.ROOT_DIR, 'data', 'file_dict')
        file_list=load_file_dict(os.path.join(file_dict_dir,"{}_file_dict.txt".format("+".join(self.package_name))))
        filename = os.path.join(file_list[index], 'Annotations', index + '.xml')
        # filename = os.path.join(self._file_dict[index], index + ".xml")
        # filename = filename.replace("JPEGImages", "Annotations")
        tree = ET.parse(filename)
        objs = tree.findall('object')
        non_diff_objs = [ obj for obj in objs if int(obj.find('difficult').text) == 0]
        objs = non_diff_objs
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            # cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            cls = self._class_to_ind[obj.find('name').text.strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(self._devkit_path,'results','VOC' + self._year,'Main',filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(index, dets[k, -1], dets[k, 0] + 1, dets[k, 1] + 1,dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(self._devkit_path,'VOC' + self._year,'Annotations','{:s}.xml')
        imagesetfile = os.path.join(self._devkit_path,'VOC' + self._year,'ImageSets','Main',self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print(('AP for {} = {:.4f}'.format(cls, ap)))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print(('Mean AP = {:.4f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print(('{:.3f}'.format(ap)))
        print(('{:.3f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc


    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed;

    embed()

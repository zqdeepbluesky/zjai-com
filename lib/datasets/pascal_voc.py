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


class pascal_voc(imdb):
    def __init__(self, image_set, year, use_diff=False):
        name = 'voc_' + year + '_' + image_set
        if use_diff:
            name += '_diff'
        imdb.__init__(self, name)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path()
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        # self._classes = ('__background__',  # always index 0
        #                  'aeroplane', 'bicycle', 'bird', 'boat',
        #                  'bottle', 'bus', 'car', 'cat', 'chair',
        #                  'cow', 'diningtable', 'dog', 'horse',
        #                  'motorbike', 'person', 'pottedplant',
        #                  'sheep', 'sofa', 'train', 'tvmonitor')

        self._classes = ('__background__',  # always index 0
                         "Tesco-sdp-dz-nyw-100g","aebs-aebsntbbt-dz-hhkw-120g","aebs-aebsytbbt-dz-hhkw-120g","ala-alabg-hz-ywjx-116g","ala-alajxbg-hz-cmwjx-116g","ala-alajxbg-hz-qtjx-116g","asm-asmnc-pz-yw-500ml","bej-bejbbt-dz-qsx-50g","bej-bejnnbbt-dz-xcnnjrsnnw-46g","bl-blht-dz-yw-6.7g","bs-bskl-gz-yw-330ml","bskl-bskl-pz-yw-600ml","ch-chgzt-dz-sjw-45g","ch-chgzt-dz-ygw-45g","ch-cht-pz-ygw-30g","cxm-cxmqkl-dz-cmxnnqkl-120g","cxm-cxmqkl-dz-cmxnxbqkl-24g","dbt-dbtnt-dz-nttg-114g","df-dfhfb-dz-nyw-168g","df-dfqkl-dz-cxhqkl-43g","df-dfqkl-dz-nxbqkl-43g","df-dfqkl-dz-shnnqkl-43g","df-dfqkl-dz-zrbdmjptgqkl-43g","df-dfqkl-dz-zrqkl-43g","dly-dlyfsrmb-dz-xcw-360g","dly-dlyfsrmb-dz-xnw-360g","dly-dlyrsj-dz-cmw-240g","dly-dlyrsj-dz-xjw-240g","fd-fdtsyl-gz-yw-330ml","fj-fjhfcb-dz-hzmw-150g","glg-glgbcbg-hz-hjqklw-48g","glg-glgbcbg-hz-qklw-48g","glg-glgblzbg-hz-cmnnw-45g","glg-glgblzbg-hz-mcxcw-45g","glg-glgbqbg-hz-cmw-55g","glg-glgbqbg-hz-mcw-50g","glg-glgbqbg-hz-nnw-60g","glg-glgbqbg-hz-qklw-60g","glg-glgllbqbg-hz-lmw-45g","gxf-gxfsjfz-dz-yw-100g","gz-gzfwgtyl-dz-nmw-400g","harbinpj-harbinpj-gz-330ml","hly-hlydhp-hz-yw-138g","hly-hlyhdy-hz-cxskw-33g","hly-hlyhdy-hz-nxqzw-33g","hly-hlyhdy-hz-xxhtw-33g","hly-hlyqklp-hz-qklw-204g","hly-hlyqtqkldg-hz-zzqklw-168g","hly-hlytlmsp-hz-yw-138g","hly-hlyytdst-dz-fqjw-70g","hly-hlyytdst-dz-ljnpw-70g","hly-hlyytdst-dz-scjyljw-45g","hn-hnwssgnyl-gz-yw-250ml","htk-tls-dz-hdw-288g","htk-tls-dz-xyw-288g","hwd-hwdfbm-tz-hxfw-84g","hwd-hwdfbm-tz-hxw-84g","hwd-hwdfbm-tz-wxnrfw-84g","hwd-hwdfbm-tz-wxnrw-84g","hy-hybfbgz-hz-xcw-200ml","hzl-hzl-gz-yw-310ml","jb-jbjyz-dz-yw-95g","jdb-jdb-pz-yw-500ml","jdb-jdblc-gz-yw-310ml","kbk-kbksp-dz-fqw-60g","kbk-kbksp-dz-skw-60g","ke-keqczyl-gz-czw-310ml","kkkl-jcnmwqs-pz-nmw-500ml","kkkl-kkkl-pz-yw-600ml","kkkl-kkkltsyl-gz-yw-330ml","ksf-ksfbg-dz-qxnmw-125g","ksf-ksfhsnrm-tz-hsnrw-64g","ksf-ksfltscnrm-tz-scnr-82.5g","ksf-ksfyrs-pz-yw-550ml","lfe-lfeyrbttgsq-hz-yrbtr-30g","lj-ljlzkxt-pz-bhw-64g","llm-llm-dz-nmcm-60g","ls-lssp-dz-mgjdyw-70g","ls-lswxsp-tz-fchgw-104g","ls-lswxsp-tz-szhsrw-104g","ls-lswxsp-tz-xnfqw-104g","ls-lswxsp-tz-yw-104g","ls-lswxsp-tz-zzkrw-104g","lzs-rnbdwhbg-hz-nlw-145g","md-mdwssyl-pz-qn-600ml","mdl-mdlbxgg-dz-bxg-80g","mms-mmsqkl-pz-mmsnnqkld-30.6g","mn-mnssr-hz-cmw-250ml","mn-mnssr-hz-yw-250ml","mn-zgl-hz-cmw-250ml","mn-zgl-hz-ygw-250ml","mz-mzxxb-pz-nnw-50g","mz-mzxxb-pz-qklw-50g","mzy-mzyglc-pz-czw-450ml","mzy-mzyglny-pz-blw-450ml","mzy-mzyglny-pz-mgw-450ml","mzy-mzyrdgl-pz-rdfw-420ml","nfsq-nfgyhhgsyl-pz-hhgs-500ml","nfsq-nfsqc-pz-xymlhc-500ml","nfsq-nfsqc-pz-yzlc-500ml","nfsq-nfsqjjydyl-pz-nmw-550ml","nfsq-nfsqsrc-pz-nmw-445ml","nfsq-nfsqyytrs-pz-yw-550ml","pp-ppfsmb-dz-nxw-400g","pp-ppfsmb-dz-xcw-400g","pp-ppfsxmb-dz-nxw-440g","qc-qckf-pz-shnt-268ml","qc-qckfjrkfyp-hz-yw-105g","qc-qcymln-tz-yw-350g","qdpj-qdpj-gz-yw-330ml","qq-qqlcgz-dz-lcw-228g","rio-rio-gz-yw-330ml","sh-shdgzyl-hz-dgw-250ml","sh-shhtc-dz-yw-500g","sh-shjrhtc-dz-jrw-240g","shj-shjhld-dz-hldw-55g","shj-shjrsyg-dz-htw-50g","shj-shjsmt-dz-cmw-40g","shj-shjst-dz-fqw-40g","shj-shjtysp-dz-fqw-80g","shj-shjtysp-dz-krw-80g","shj-shjxxp-dz-xxw-40g","shj-shjycq-dz-ycw-40g","slj-sljqkl-dz-hsjx-35g","tdyh-tdyhpgc-gz-pg-330ml","ty-hzy-pz-gw-500ml","ty-tyhsnrm-tz-nr-105g","ty-tyxmtx-pz-lpqnhc-480ml","wl-wldmj-dz-lw-106g","wlj-wljlc-gz-yw-310ml","wlj-wljlc-hz-yw-250ml","wlj-wljlc-pz-yw-500ml","wq-wqaljm-dz-al-50g","wt-wtcyl-gz-nm-310ml","wt-wtnmc-pz-nm-500ml","wtn-wtnywdn-hz-yw-250ml","wwsp-wwhshtmb-dz-htw-118g","wwsp-wwxb-dz-yw-84g","wwsp-wwxxs-dz-yw-60g","wwsp-wznn-gz-yw-145ml","wwsp-wznn-hz-yw-125ml","wwsp-wzxmt-dz-nnw-210g","xb-xbtsyl-gz-yw-330ml","xhpj-xhqspj-gz-yw-330ml","xpp-xppnc-bz-cmw-80g","xpp-xppnc-bz-mxw-80g","xpp-xppnc-bz-xyw-80g","xpp-xppnc-bz-yw-80g","yb-ybcjs-pz-yw-555ml","yd-ydwtkxt-hz-rdsgw-32g","yd-ydwtkxt-pz-blbhw-56g","yd-ydwtkxt-pz-qscmw-56g","yd-ydwtkxt-pz-qsxgw-56g","yd-ydwtkxt-pz-qxlmw-56g","yd-ydwtkxt-pz-xcbhw-56g","yd-ydwtkxt-pz-xnmgw-56g","yj-pjfz-dz-sjw-100g","yl-ylcnn-hz-yw-250ml","yl-ylgldnn-hz-heiguw-250ml","yl-ylgldnn-hz-hongguw-250ml","yl-ylhzdhmbbz-gz-hm-280g","yl-ylqqxetcznn-hz-jgx-125ml","yl-ylqqxetcznn-hz-qcx-125ml","yl-ylshwrtnn-hz-qzx-220ml","yl-ylysr-hz-cmw-250ml","yl-ylysr-hz-yw-250ml","ys-zzyspyz-gz-yw-245ml","ys-zzyspyz-hz-yw-245ml","yy-yylght-gz-ht-240ml")
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': use_diff,
                       'matlab_eval': False,
                       'rpn_file': None}

        assert os.path.exists(self._devkit_path), 'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
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
        assert os.path.exists(filename), 'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
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

            if x1 > 5000:
                print('a')
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VOC' + self._year,
            'Main',
            filename)
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
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
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
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
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

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print(('Running:\n{}'.format(cmd)))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 7/6/2018 17:26 AM
# @Author : jaykky
# @File : zjai_6_comparison.py
# @Software: ZJ_AI
#此程序是用于比对模型的识别成果和真实情况，并得到精确率和召回率的模型指标。
#主要方法是对比模型识别出来的object和xml文件的信息
#属于测试fasterrcnn的模型性能
#输入：模型文件路径，测试图像路径，
#输出：精确率和召回率
# =========================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')
# CLASSES=('__background__','wl-wldmj-dz-lw-106g', 'llm-llm-dz-nmcm-60g', 'ksf-ksfbg-dz-qxnmw-125g', 'htk-tls-dz-hd-288g',
#                    'ty-hzy-pz-gw-500ml', 'ls-lssp-dz-mgjdyw-70g', 'bl-blht-dz-yw-6.7g', 'yj-pjfz-dz-sjw-100g',
#                    'yy-yylght-gz-ht-240ml', 'qc-qckf-pz-shnt-268ml', 'tdyh-tdyhpgc-gz-pg-330ml',
#                    'ty-tyxmtx-pz-lpqnhc-480ml', 'yl-ylhzdhmbbz-gz-hm-280g', 'ksf-ksfltscnrm-tz-scnr-82.5g',
#                    'jdb-jdblc-gz-yw-310ml', 'ty-tyhsnrm-tz-nr-105g', 'mn-zgl-hz-cmw-250ml', 'yl-ylcnn-hz-yw-250ml',
#                    'asm-asmnc-pz-yw-500ml', 'glg-glgblzbg-hz-mcxcw-45g', 'lzs-rnbdwhbg-hz-nlw-145g',
#                    'wq-wqaljm-dz-al-50g', 'ys-zzyspyz-gz-yw-245ml', 'wt-wtcyl-gz-nm-310ml',
#                    'lfe-lfeyrbttgsq-hz-yrbtr-30g', 'wwsp-wwxxs-dz-yw-60g', 'ala-alabg-hz-ywjx-116g',
#                    'mdl-mdlbxgg-dz-bxg-80g', 'nfsq-nfsqc-pz-yzlc-500ml', 'bs-bskl-gz-yw-330ml',
#                    'nfsq-nfsqc-pz-xymlhc-500ml', 'hzl-hzl-gz-yw-310ml', 'wwsp-wznn-hz-yw-125ml',
#                    'wtn-wtnywdn-hz-yw-250ml', 'hwd-hwdfbm-tz-hxw-75g', 'kkkl-jcnmwqs-pz-nmw-500ml',
#                    'kkkl-kkkl-pz-yw-600ml', 'wlj-wljlc-pz-yw-500ml', 'jb-jbjyz-dz-yw-95g',
#                    'yd-ydmtcqscm-pz-cmw-56g', 'wlj-wljlc-hz-yw-250ml', 'hwd-hwdfbm-tz-wxnrfw-84g')
CLASSES=('__background__','rio-rio-gz-yw-330ml', 'hzl-hzl-gz-yw-310ml', 'harbinpj-harbinpj-gz-330ml',
                   'df-dfqkl-dz-zrbdmjptgqkl-43g', 'shj-shjtysp-dz-fqw-80g', 'glg-glgbcbg-hz-qklw-48g',
                   'yd-ydwtkxt-pz-qscmw-56g', 'ty-tyhsnrm-tz-nr-105g', 'hly-hlydhp-hz-yw-138g', 'nfsq-nfsqsrc-pz-nmw-445ml',
                   'glg-glgbqbg-hz-nnw-60g', 'jdb-jdblc-gz-yw-310ml', 'wl-wldmj-dz-lw-106g', 'gxf-gxfsjfz-dz-yw-100g',
                   'fj-fjhfcb-dz-hzmw-150g', 'glg-glgblzbg-hz-mcxcw-45g', 'ly-hlydhp-hz-yw-138g', 'ksf-ksfyrs-pz-yw-550ml',
                   'wwsp-wwxb-dz-yw-84g', 'dbt-dbtnt-dz-nttg-114g', 'pp-ppfsxmb-dz-nxw-440g', 'glg-glgbqbg-hz-mcw-50g',
                   'cxm-cxmqkl-dz-cmxnxbqkl-24g', 'ls-lswxsp-tz-zzkrw-104g', 'hwd-hwdfbm-tz-hxfw-84g',
                   'xpp-xppnc-bz-xyw-80g', 'yl-ylshwrtnn-hz-qzx-220ml', 'hly-hlyhdy-hz-xxhtw-33g',
                   'xpp-xppnc-bz-mxw-80g', 'sh-shdgzyl-hz-dgw-250ml', 'gz-gzfwgtyl-dz-nmw-400g',
                   'mzy-mzyglny-pz-mgw-450ml', 'nfsq-nfsqyytrs-pz-yw-550ml', 'ala-alabg-hz-ywjx-116g',
                   'bskl-bskl-pz-yw-600ml', 'nfsq-nfsqjjydyl-pz-nmw-550ml', 'kkkl-kkkltsyl-gz-yw-330ml',
                   'yl-ylysr-hz-yw-250ml', 'xb-xbtsyl-gz-yw-330ml', 'kbk-kbksp-dz-skw-60g', 'ys-zzyspyz-gz-yw-245ml',
                   'qdpj-qdpj-gz-yw-330ml', 'hly-hlyytdst-dz-ljnpw-70g', 'wwsp-wznn-hz-yw-125ml', 'mn-mnssr-hz-yw-250ml',
                   'hly-hlyqklp-hz-qklw-204g', 'hwd-hwdfbm-tz-hxw-84g', 'hly-hlyytdst-dz-fqjw-70g',
                   'yd-ydwtkxt-hz-rdsgw-32g', 'xhpj-xhqspj-gz-yw-330ml', 'hly-hlytlmsp-hz-yw-138g',
                   'yb-ybcjs-pz-yw-555ml', 'yl-ylqqxetcznn-hz-qcx-125ml', 'mn-mnssr-hz-cmw-250ml',
                   'yd-ydwtkxt-pz-xnmgw-56g', 'mdl-mdlbxgg-dz-bxg-80g', 'yl-ylgldnn-hz-heiguw-250ml',
                   'glg-glgbqbg-hz-qklw-60g', 'yl-ylgldnn-hz-hongguw-250ml', 'ch-chgzt-dz-ygw-45g',
                   'yl-ylysr-hz-cmw-250ml', 'glg-glgbcbg-hz-hjqklw-48g', 'nfsq-nfsqc-pz-yzlc-500ml',
                   'nfsq-nfgyhhgsyl-pz-hhgs-500ml', 'slj-sljqkl-dz-hsjx-35g', 'mz-mzxxb-pz-nnw-50g',
                   'pp-ppfsmb-dz-xcw-400g', 'df-dfqkl-dz-shnnqkl-43g', 'bej-bejnnbbt-dz-xcnnjrsnnw-46g',
                   'shj-shjxxp-dz-xxw-40g', 'aebs-aebsntbbt-dz-hhkw-120g', 'mzy-mzyglc-pz-czw-450ml',
                   'yd-ydwtkxt-pz-blbhw-56g', 'qq-qqlcgz-dz-lcw-228g', 'df-dfqkl-dz-cxhqkl-43g', 'glg-glgllbqbg-hz-lmw-45g',
                   'mz-mzxxb-pz-qklw-50g', 'lfe-lfeyrbttgsq-hz-yrbtr-30g', 'yy-yylght-gz-ht-240ml',
                   'glg-glgblzbg-hz-cmnnw-45g', 'glg-glgbqbg-hz-cmw-55g', 'tdyh-tdyhpgc-gz-pg-330ml',
                   'yd-ydwtkxt-pz-xcbhw-56g', 'hn-hnwssgnyl-gz-yw-250ml', 'fd-fdtsyl-gz-yw-330ml', 'mn-zgl-hz-ygw-250ml',
                   'df-dfqkl-dz-zrqkl-43g', 'hy-hybfbgz-hz-xcw-200ml', 'wwsp-wznn-gz-yw-145ml', 'ys-zzyspyz-hz-yw-245ml',
                   'yl-ylcnn-hz-yw-250ml', 'ke-keqczyl-gz-czw-310ml', 'ala-alajxbg-hz-qtjx-116g', 'yd-ydwtkxt-pz-qsxgw-56g',
                   'ch-chgzt-dz-sjw-45g', 'wwsp-wzxmt-dz-nnw-210g', 'yl-ylqqxetcznn-hz-jgx-125ml', 'lj-ljlzkxt-pz-bhw-64g',
                   'xpp-xppnc-bz-cmw-80g', 'jdb-jdb-pz-yw-500ml', 'wlj-wljlc-hz-yw-250ml', 'wtn-wtnywdn-hz-yw-250ml',
                   'df-dfqkl-dz-nxbqkl-43g', 'wwsp-wwxxs-dz-yw-60g', 'nfsq-nfsqc-pz-xymlhc-500ml',
                   'ls-lswxsp-tz-szhsrw-104g', 'ty-tyxmtx-pz-lpqnhc-480ml', 'ksf-ksfbg-dz-qxnmw-125g', 'ty-hzy-pz-gw-500ml',
                   'hwd-hwdfbm-tz-wxnrw-84g', 'ls-lssp-dz-mgjdyw-70g', 'ksf-ksfhsnrm-tz-hsnrw-64g',
                   'mzy-mzyglny-pz-blw-450ml', 'wt-wtnmc-pz-nm-500ml', 'bl-blht-dz-yw-6.7g', 'qc-qckfjrkfyp-hz-yw-105g',
                   'df-dfhfb-dz-nyw-168g', 'Tesco-sdp-dz-nyw-100g', 'kbk-kbksp-dz-fqw-60g', 'md-mdwssyl-pz-qn-600ml',
                   'wlj-wljlc-pz-yw-500ml', 'kkkl-kkkl-pz-yw-600ml', 'jb-jbjyz-dz-yw-95g', 'bs-bskl-gz-yw-330ml',
                   'qc-qckf-pz-shnt-268ml', 'xpp-xppnc-bz-yw-80g', 'ls-lswxsp-tz-yw-104g', 'hwd-hwdfbm-tz-wxnrfw-84g',
                   'ksf-ksfltscnrm-tz-scnr-82.5g', 'sh-shjrhtc-dz-jrw-240g', 'llm-llm-dz-nmcm-60g', 'wt-wtcyl-gz-nm-310ml',
                   'sh-shhtc-dz-yw-500g', 'shj-shjsmt-dz-cmw-40g', 'dly-dlyfsrmb-dz-xcw-360g', 'yl-ylhzdhmbbz-gz-hm-280g',
                   'htk-tls-dz-xyw-288g', 'hly-hlyqtqkldg-hz-zzqklw-168g', 'mms-mmsqkl-pz-mmsnnqkld-30.6g',
                   'wq-wqaljm-dz-al-50g', 'pp-ppfsmb-dz-nxw-400g', 'dly-dlyrsj-dz-xjw-240g', 'yj-pjfz-dz-sjw-100g',
                   'qc-qcymln-tz-yw-350g', 'dly-dlyrsj-dz-cmw-240g', 'wlj-wljlc-gz-yw-310ml', 'shj-shjtysp-dz-krw-80g',
                   'dly-dlyfsrmb-dz-xnw-360g', 'ch-cht-pz-ygw-30g', 'hly-hlyytdst-dz-scjyljw-45g', 'hly-hlyhdy-hz-cxskw-33g',
                   'lzs-rnbdwhbg-hz-nlw-145g', 'mn-zgl-hz-cmw-250ml', 'kkkl-jcnmwqs-pz-nmw-500ml', 'asm-asmnc-pz-yw-500ml',
                   'shj-shjst-dz-fqw-40g', 'yd-ydwtkxt-pz-qxlmw-56g', 'wwsp-wwhshtmb-dz-htw-118g', 'htk-tls-dz-hdw-288g',
                   'shj-shjycq-dz-ycw-40g', 'mzy-mzyrdgl-pz-rdfw-420ml', 'shj-shjhld-dz-hldw-55g', 'bej-bejbbt-dz-qsx-50g',
                   'ls-lswxsp-tz-fchgw-104g', 'ls-lswxsp-tz-xnfqw-104g', 'aebs-aebsytbbt-dz-hhkw-120g',
                   'cxm-cxmqkl-dz-cmxnnqkl-120g', 'shj-shjrsyg-dz-htw-50g', 'hly-hlyhdy-hz-nxqzw-33g',
                   'ala-alajxbg-hz-cmwjx-116g')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    count = len(inds)
    ax.add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1], fill=False,
                        edgecolor='red', linewidth=3.5)
        )
    ax.text(bbox[0], bbox[1] - 2,
            '{:s} {:.3f}'.format(class_name, score),
            bbox=dict(facecolor='blue', alpha=0.5),
            fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(sess, net, image_name,classDict):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image

    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        # vis_detections(im, cls, dets, thresh=CONF_THRESH)
        count=len(np.where(dets[:, -1] >= CONF_THRESH)[0])
        if cls in list(classDict.keys()):
            classDict[cls]+=count
        else:
            classDict[cls]=count
    return classDict

def getImageLabelNum(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image

    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    countNum=0
    labelList=[]
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        # vis_detections(im, cls, dets, thresh=CONF_THRESH)
        count=len(np.where(dets[:, -1] >= CONF_THRESH)[0])
        if count!=0:
            countNum += count
            labelList.append(cls)
    return countNum,labelList

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

def getTestImagePath(testIamgeTxtDir,imageDir):
    dirs=[]
    with open(testIamgeTxtDir,"r") as f:
        lineList=f.readlines()
        for line in lineList:
            dirs.append(imageDir+"/"+line.replace("\n",'')+".jpg")
    return dirs

def getXmlLabelNum(xmlPath):
    et = ET.parse(xmlPath)
    element = et.getroot()
    element_objs = element.findall('object')
    count=len(element_objs)
    labelList=[]
    for element_obj in element_objs:
        node = element_obj.find('name')
        label=node.text
        labelList.append(label)
    return count,labelList

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])

    #tfmodel="/home/hyl/data/ljk/project/2-shopDetect/tf-faster-rcnn-master/data/"+tfmodel
    # tfmodel ='/home/hyl/data/ljk/project/2-shopDetect/tf-faster-rcnn-master/data/output/res101/voc_2007_trainval+voc_2012_trainval/default/res101_faster_rcnn_iter_70000.ckpt'
    tfmodel ="/home/hyl/data/ljk/project/2-shopDetect/tf-faster-rcnn-master/output/res101/voc_2007_trainval/default/res101_faster_rcnn_iter_70000.ckpt"
    print(tfmodel)
    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 171,
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    testIamgeTxtDir="/home/hyl/data/ljk/project/2-shopDetect/tf-faster-rcnn-master/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt"
    imageDir='/home/hyl/data/ljk/project/2-shopDetect/tf-faster-rcnn-master/data/VOCdevkit2007/VOC2007/JPEGImages'
    dirs=getTestImagePath(testIamgeTxtDir,imageDir)
    classDict={}
    d_count=0
    tp_sum=0
    fp_sum=0
    fn_sum=0
    d_sum=0
    t_sum=0
    for dir in dirs:
        fp = 0
        tp = 0
        fn = 0
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # print('Demo for data/demo/{}'.format(dir))
        # classDict=demo(sess, net, dir,classDict)
        d_labelNum,d_labelList=getImageLabelNum(sess,net,dir)
        # d_labelNum, d_labelList =getXmlLabelNum(dir.replace("JPEGImages","Annotations")[:-4]+".xml")
        t_labelNum,t_labelList=getXmlLabelNum(dir.replace("JPEGImages","Annotations")[:-4]+".xml")
        for d_label in d_labelList:
            if d_label in t_labelList:
                tp+=1
            else:
                fp+=1
        fn=t_labelNum-tp
        tp_sum+=tp
        fp_sum+=fp
        fn_sum+=fn
        d_sum+=d_labelNum
        t_sum+=t_labelNum
        print(tp,fp,fn,d_labelNum,t_labelNum)
    recall=(tp_sum)/(tp_sum+fn_sum)
    precsion=(tp_sum)/(tp_sum+fp_sum)
    print(recall,precsion)
    print(tp_sum,fp_sum,fn_sum)
    print(d_sum,t_sum)




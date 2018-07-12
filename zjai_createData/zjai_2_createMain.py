#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 7/6/2018 17:26 AM
# @Author : jaykky
# @File : zjai_2_createMain.py
# @Software: ZJ_AI
#此程序是用于创建fasterrcnn的Main文件夹。
#属于构造fasterrcnn的数据集的步骤二
#输入：父文件夹路径
#输出：用于 训练评估和测试的图像名称列表 的txt文件
# =========================================================


import os
import os.path as osp
import random
from data_processing.utils.io_utils import *
from zjai_createData import check_exist
from zjai_createData.check_exist import getAllFile

def _create_Main(dataDirs,fileList,scale):
    '''
    create the trainval.txt and test.txt for train.
    trainval data : test data = 5:1
    :param path:
    :return:
    '''
    trainval_images = []
    test_images = []
    mkdir(osp.join(dataDirs,"ImageSets","Main"))
    for i in range(len(fileList)//scale, len(fileList)):
        s = fileList[i]
        if dataDirs[-1]=="/":
            s=s.replace(dataDirs,"")
        else:
            s=s.replace(dataDirs + "/", "")
        trainval_images.append(s.split('.')[0] + '\n')

    for i in range(len(fileList)//scale):
        s = fileList[i]
        if dataDirs[-1]=="/":
            s=s.replace(dataDirs,"")
        else:
            s=s.replace(dataDirs + "/", "")
        test_images.append(s.split('.')[0] + '\n')

    with open(dataDirs+'/ImageSets/Main/trainval.txt','w+') as f:
        f.writelines(trainval_images)
        print("{}, numbers:{}".format(dataDirs + '/trainval.txt', len(trainval_images)))
    with open(dataDirs+'/ImageSets/Main/test.txt','w+') as f:
        f.writelines(test_images)
        print("{}, numbers:{}".format(dataDirs + '/test.txt', len(test_images)))

    print('total: {}'.format(len(fileList)))
    print('step: {}'.format(len(trainval_images)//2+1))

if __name__=="__main__":
    # root_dir = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    root_dir = "/home/hyl/data/ljk/project/2-shopDetect/tf-faster-rcnn-master"
    dataDirs = osp.join(root_dir, 'data', 'train_data')
    scale = 9
    fileList=getAllFile(dataDirs,fileType="jpg")
    _create_Main(dataDirs,fileList,scale)

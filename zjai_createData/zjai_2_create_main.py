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


import os.path as osp
from morelib.utils.io_utils import *
from zjai_createData.check_exist import get_all_file

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
    with open(dataDirs+'/ImageSets/Main/trainval_test.txt','w+') as f:
        f.writelines(trainval_images)
        f.writelines(test_images)

    print('total: {}'.format(len(fileList)))
    print('step: {}'.format(len(trainval_images)//2+1))

def _create_Main_new(dataDirs,fileList,scale):
    '''
    create the trainval.txt and test.txt for train.
    trainval data : test data = 5:1
    :param path:
    :return:
    '''
    trainval_images = []
    test_images = []
    fileDict={}
    mkdir(osp.join(dataDirs,"ImageSets","Main"))
    for i in range(len(fileList)//scale, len(fileList)):
        s = fileList[i]
        parentDir=osp.dirname(s)
        filename=osp.splitext(s)[0].replace(parentDir,"").replace("/","")
        trainval_images.append(filename + '\n')
        fileDict[filename]=parentDir

    for i in range(len(fileList)//scale):
        s = fileList[i]
        parentDir = osp.dirname(s)
        filename = osp.splitext(s)[0].replace(parentDir, "").replace("/", "")
        test_images.append(filename + '\n')
        fileDict[filename] = parentDir

    with open(dataDirs+'/ImageSets/Main/trainval.txt','w+') as f:
        f.writelines(trainval_images)
        print("{}, numbers:{}".format(dataDirs + '/trainval.txt', len(trainval_images)))
    with open(dataDirs+'/ImageSets/Main/test.txt','w+') as f:
        f.writelines(test_images)
        print("{}, numbers:{}".format(dataDirs + '/test.txt', len(test_images)))
    with open(dataDirs+'/ImageSets/Main/trainval_test.txt','w+') as f:
        f.writelines(trainval_images)
        f.writelines(test_images)
    with open(dataDirs+'/ImageSets/Main/filedict.txt','w+') as f:
        for key in fileDict.keys():
            f.write("{}|{}\n".format(key,fileDict[key]))
    print('total: {}'.format(len(fileList)))
    print('step: {}'.format(len(trainval_images)//2+1))

def create_package_main(data_dir,scale):
    fileList = get_all_file(data_dir, fileType="jpg")
    _create_Main_new(data_dir, fileList, scale)

if __name__=="__main__":
    root_dir = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    dataDirs = osp.join(root_dir, 'data', 'predict_data',"test_data-2018-07-24")
    dataDirs = '/home/hyl/data/ljk/github-pro/zjai-com/data/train_data/train_data-2018-08-15_resize'
    scale = 9
    fileList=get_all_file(dataDirs,fileType="jpg")
    _create_Main_new(dataDirs,fileList,scale)

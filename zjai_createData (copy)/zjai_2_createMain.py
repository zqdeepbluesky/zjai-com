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

from data_processing import create_Main
from data_processing import create_all_Main
import sys
import os
import random

pwd=sys.path[0]
# srcPath="/home/hyl/data/data-lyl"
parentDir=os.path.dirname(pwd)
dataSetDir=os.path.join(parentDir,"dataSet")
# dataSetDir="/home/hyl/data/data-lyl/test_data-2018-06-15"
dataSetDir="/home/hyl/data/data-lyl/2018-07-09_test"
scale=9

def createBasicMain(dataSetPath,scale):
    '''
    函数调用data_processing的函数
    :param dataSetPath:
    :param scale:划分比例
    :return:
    '''
    # dataSetPath = getPath()
    dirs=[]
    dirs.append(dataSetPath)
    create_Main.create_subs_Main_new(dirs,scale)

createBasicMain(dataSetDir,scale)
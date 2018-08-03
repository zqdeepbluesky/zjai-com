#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 7/6/2018 17:26 AM
# @Author : jaykky
# @File : ajai_1_copyCollect.py
# @Software: ZJ_AI
#此程序是拿来把多个子文件夹合成一个文件夹。
#属于构造fasterrcnn的数据集的步骤一
#输入：父文件夹路径
#输出：完整的Annotation、JPEGImages文件夹
# =========================================================

import os
from morelib.utils import io_utils
import sys

def mkdir_dir(dataSetDir):
    '''
    在输入路径下创建文件结构：Annotation、ImageSets/Main、JPEGImages
    :param dataSetDir: 构造文件结构的父路径
    :return:
    '''
    annotPath = os.path.join(dataSetDir, "Annotations")
    mainPath = os.path.join(dataSetDir, "ImageSets/Main")
    JPEGPath = os.path.join(dataSetDir, "JPEGImages")
    io_utils.mkdir(annotPath)
    io_utils.mkdir(mainPath)
    io_utils.mkdir(JPEGPath)
    return annotPath,mainPath,JPEGPath

def copy_xml(srcPath,distPath):
    '''
    函数主要用于将srcPath文件夹下的xml文件复制到distPath/Annotations文件夹下
    :param srcPath: 数据源
    :param distPath:
    :return:
    '''
    fileType="xml"
    many2one(srcPath,fileType,distPath)
def copy_jpg(srcPath,distPath):
    '''
    函数主要用于将srcPath文件夹下的所有JPG文件复制到distPath/JPEGImages文件夹下
    :param srcPath:数据源
    :param distPath:
    :return:
    '''
    fileType="jpg"
    many2one(srcPath,fileType,distPath)
def many2one(parent_dir,type,distPath):
    '''
    核心函数，将parent_dir文件夹下所有后缀名为type的文件复制到distParh下
    :param parent_dir: 父路径
    :param type: 文件后缀名
    :param distPath: 目标数据位置
    :return:
    '''
    count=0
    for parent, dirnames, filenames in os.walk(parent_dir):
        for filename in filenames:
            fileName=os.path.join(parent,filename)
            if fileName[-3:]==type:
                #print(fileName)
                distFile=os.path.join(distPath,filename)
                if os.path.exists(distFile):
                    print(fileName,"error")
                io_utils.copy(fileName,distFile )
                count+=1
    print("{} files have copy ".format(type)+"{} file".format(count))

pwd=sys.path[0]
srcPath="/home/hyl/data/data-lyl"
# srcPath="/home/hyl/data/data-lyl/test_data-2018-06-15"
parentDir=os.path.dirname(pwd)
dataSetDir=os.path.join(parentDir,"dataSet")

annotPath, mainPath, JPEGPath=mkdir_dir(dataSetDir)
copy_xml(srcPath,annotPath)
copy_jpg(srcPath,JPEGPath)



#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 4/23/2018 17:26 AM
# @Author : CarrieChen
# @File : after_label.py
# @Software: ZJ_AI
#此程序是拿来对预测后的图片进行分包
#输入：预测后的图片
#输出：分好包的压缩包，每个压缩包中有Annotations文件夹和JPEGImages文件夹。
# =========================================================

import os
import zipfile

import cv2
import io
import utils.io_utils

import numpy as np
import datetime
import argparse

def create_zip(src_dir):
    # root_home = os.path.dirname(args.parent_dir)
    # JPEGImages_dir = os.path.join(args.parent_dir,'-'.format(idx) ,'JPEGImages\\')
    # Annotations_dir = os.path.join(args.parent_dir, '-'.format(idx) ,'Annotations\\')
    # io_utils.mkdir(JPEGImages_dir)
    # io_utils.mkdir(Annotations_dir)
    # io_utils.remove_all(JPEGImages_dir)
    # io_utils.remove_all(Annotations_dir)

    JPEGImages_dir = os.path.join(src_dir,'JPEGImages\\')
    Annotations_dir = os.path.join(src_dir,'Annotations\\')
    idx_j = 0
    folder_j = 0
    idx_a = 0
    folder_a = 0
    temp_dir = os.path.join(src_dir, str_date[0:4] + '-' + str_date[4:6] + '-' + str_date[6:8])

    for s in os.listdir(JPEGImages_dir):  #JPEGImages分包
        if idx_j % 200 == 0: #每200张图片操作一次,从第0张开始
            if folder_a >= 1:
                parent = temp_dir+'-{}'.format(folder_j-1)
                parent_zip = temp_dir+'-{}.zip'.format(folder_j-1)
                zip_dir(parent, parent_zip)

            childJPEGImages_dir = os.path.join(temp_dir + '-{}'.format(folder_j), 'JPEGImages\\')
            io_utils.mkdir(childJPEGImages_dir)
            io_utils.remove_all(childJPEGImages_dir)  #确保目标文件夹是空白的
            folder_j+=1
        idx_j+=1
        file = os.path.join(JPEGImages_dir, s)
        io_utils.copy(file, childJPEGImages_dir)

    #
    parent = temp_dir + '-{}'.format(folder_j-1)
    parent_zip = temp_dir + '-{}.zip'.format(folder_j-1)
    zip_dir(parent, parent_zip)

    for s in os.listdir(Annotations_dir):  # JPEGImages分包
        if idx_a % 200 == 0:  # 每200张图片操作一次,从第0张开始
            if folder_a >= 1:
                parent = temp_dir + '-{}'.format(folder_a - 1)
                parent_zip = temp_dir + '-{}.zip'.format(folder_a - 1)
                zip_dir(parent, parent_zip)
            childAnnotations_dir = os.path.join(temp_dir + '-{}'.format(folder_a), 'Annotations\\')
            io_utils.mkdir(childAnnotations_dir)
            io_utils.remove_all(childAnnotations_dir)
            folder_a += 1
        idx_a += 1
        file = os.path.join(Annotations_dir, s)
        io_utils.copy(file, childAnnotations_dir)
        #
    parent = temp_dir + '-{}'.format(folder_a - 1)
    parent_zip = temp_dir + '-{}.zip'.format(folder_a - 1)
    zip_dir(parent, parent_zip)

def zip_dir(file_path,zfile_path):
    '''
    function:压缩
    params:
        file_path:要压缩的文件路径,可以是文件夹
        zfile_path:解压缩路径
    description:可以在python2执行
    '''
    filelist = []  #待压缩文件列表
    if os.path.isfile(file_path): #如果path是一个存在的文件，返回True。否则返回False。
        filelist.append(file_path)
    else :
        for root, dirs, files in os.walk(file_path):  #os.walk() 方法用于通过在目录树中游走输出在目录中的文件名
            for name in files:
                filelist.append(os.path.join(root, name))
                print('joined:',os.path.join(root, name),dirs)

    zf = zipfile.ZipFile(zfile_path, "w", zipfile.zlib.DEFLATED)
    for tar in filelist:   #？？？
        arcname = tar[len(file_path):]
        print(arcname,tar)
        zf.write(tar,arcname)
    zf.close()

str_date = '{year}{month}{day}'.format(year='2018', month='04', day='18')  #改这里，日期
parser = argparse.ArgumentParser(description='Get the data info')
parser.add_argument('-p', '--parent_dir',help='the parent folder of image', default='D:\\Predictions\\'+str_date[0:4]+'-'+str_date[4:6]+'-'+str_date[6:8])  #windows系统下用\\
parser.add_argument('-d', '--folder_name',help='the origin folder of image', default='origin')
args = parser.parse_args()

if __name__  == '__main__':
    if args.parent_dir and args.folder_name:
        create_zip(args.parent_dir)

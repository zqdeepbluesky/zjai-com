#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 3/29/2018 9:33 AM 
# @Author : sunyonghai 
# @File : before_label.py 
# @Software: ZJ_AI
#此程序是拿来做图片预处理（旋转90度和重命名）
#输入：拍照回来的原始图片
#输出：分好包的压缩包，每个压缩包中有Annotations文件夹(含test.txt，记得除去)和JPEGImages文件夹。
# =========================================================
import os
import zipfile

import cv2
import io
from data_processing.utils import io_utils

import numpy as np
import datetime
import argparse

def create_origin(parent_dir,origin_dir): #创建一个origin文件夹
    origin_pic_dir = os.path.join(parent_dir,origin_dir)
    io_utils.mkdir(origin_pic_dir)
    for s in os.listdir(parent_dir):
        file = os.path.join(parent_dir, s)
        io_utils.move(file, origin_pic_dir)


def rot90(parent_dir, image_dir): #旋转90度
    imgs_path = os.path.join(parent_dir, image_dir)
    imgs_out_path = os.path.join(parent_dir, '{}{}'.format(image_dir, '_with_rot90'))
    io_utils.delete_file_folder(imgs_out_path)
    io_utils.mkdir(imgs_out_path)
    #print(imgs_out_path)


    images = [os.path.join(imgs_path, s) for s in os.listdir(imgs_path)]
    for image_file in images:
        try:
            img = cv2.imread(image_file)
            width = img.shape[0]
            height = img.shape[1]
            if width > height:
                image = np.array(np.rot90(img, 1))
                image = image.copy()
            else:
                image = img
            name = image_file.split('/')[-1]
            save_path = os.path.join(imgs_out_path, name)

            # don't need resize
            # image = cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0]*0.5)), interpolation=cv2.INTER_CUBIC)
            # print('resize:{}'.format(image.shape))
            cv2.imwrite(save_path, image)

            #print(save_path)
        except Exception as e:
            print('Exception in pascal_voc_parser: {}'.format(e))
            continue

    return imgs_out_path

def rename_image(parent_dir, image_dir_name): #重命名
    # image_dir_name = 'JPEGImages'
    data_dir = os.path.join(parent_dir, image_dir_name)
    data_rename_dir = os.path.join(parent_dir, '{}_rename'.format(image_dir_name))

    io_utils.delete_file_folder(data_rename_dir)
    io_utils.mkdir(data_rename_dir)
    prefix = 'train'
    idx = 1000  #起始编码id
    cur_date = datetime.datetime.now()
    # str_date = '{year}{month}{day}'.format(year=cur_date.year, month=cur_date.month, day=cur_date.day)
    for s in os.listdir(data_dir):
        old = os.path.join(data_dir, s)
        new = os.path.join(data_rename_dir, '{}_{}_{}.jpg'.format(prefix, str_date[0:4]+'-'+str_date[4:6]+'-'+str_date[6:8], idx))
        io_utils.copy(old,new)
        idx = idx+1

    return data_rename_dir

def copy_to_JPEGImages(src_dir): #把处理好的图片放到JPEGImages
    target_dir = os.path.join(args.parent_dir, 'JPEGImages/')
    io_utils.mkdir(target_dir)
    io_utils.remove_all(target_dir)

    for s in os.listdir(src_dir):
        file = os.path.join(src_dir, s)
        io_utils.copy(file, target_dir)


def create_subs(src_dir):
    # root_home = os.path.dirname(args.parent_dir)

    # JPEGImages_dir = os.path.join(args.parent_dir,'-'.format(idx) ,'JPEGImages\\')
    # Annotations_dir = os.path.join(args.parent_dir, '-'.format(idx) ,'Annotations\\')
    # io_utils.mkdir(JPEGImages_dir)
    # io_utils.mkdir(Annotations_dir)
    # io_utils.remove_all(JPEGImages_dir)
    # io_utils.remove_all(Annotations_dir)

    idx = 0
    folder = 10
    for s in os.listdir(src_dir):
        if idx % 200 == 0:
            JPEGImages_dir = os.path.join(args.parent_dir + '-{}'.format(folder), 'JPEGImages/')
            Annotations_dir = os.path.join(args.parent_dir + '-{}'.format(folder), 'Annotations/')
            io_utils.mkdir(JPEGImages_dir)
            io_utils.mkdir(Annotations_dir)
            io_utils.remove_all(JPEGImages_dir)
            io_utils.remove_all(Annotations_dir)
            folder += 1
        idx += 1
        file = os.path.join(src_dir, s)
        io_utils.copy(file, JPEGImages_dir)



def create_zip(src_dir,parent_dir): #分包加压缩
    # root_home = os.path.dirname(args.parent_dir)
    # JPEGImages_dir = os.path.join(args.parent_dir,'-'.format(idx) ,'JPEGImages\\')
    # Annotations_dir = os.path.join(args.parent_dir, '-'.format(idx) ,'Annotations\\')
    # io_utils.mkdir(JPEGImages_dir)
    # io_utils.mkdir(Annotations_dir)
    # io_utils.remove_all(JPEGImages_dir)
    # io_utils.remove_all(Annotations_dir)
    idx = 0
    folder = 0

    for s in os.listdir(src_dir):
        if idx % 200 == 0: #每200张图片操作一次,从第0张开始
            if folder >= 1:
                parent = temp_dir+'-{}'.format(folder-1)
                parent_zip = temp_dir+'-{}.zip'.format(folder-1)
                zip_dir(parent, parent_zip)

            temp_dir=os.path.join(parent_dir,str_date[0:4]+'-'+str_date[4:6]+'-'+str_date[6:8])
            JPEGImages_dir = os.path.join(temp_dir + '-{}'.format(folder), 'JPEGImages/')
            #Annotations_dir = os.path.join(temp_dir + '-{}'.format(folder), 'Annotations\\')
            io_utils.mkdir(JPEGImages_dir)
            #io_utils.mkdir(Annotations_dir)
            io_utils.remove_all(JPEGImages_dir)  #确保目标文件夹是空白的
            #io_utils.remove_all(Annotations_dir)
            folder+=1
        idx+=1
        file = os.path.join(src_dir, s)
        io_utils.copy(file, JPEGImages_dir)
        #full_path = Annotations_dir + '\\' + 'test' + '.txt'  # 加入了一个txt
        #open(full_path, 'w')



    #
    parent = temp_dir + '-{}'.format(folder-1)
    parent_zip = temp_dir + '-{}.zip'.format(folder-1)
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
                #print('joined:',os.path.join(root, name),dirs)

    zf = zipfile.ZipFile(zfile_path, "w", zipfile.zlib.DEFLATED)
    for tar in filelist:   #？？？
        arcname = tar[len(file_path):]
        #print(arcname,tar)
        zf.write(tar,arcname)
    zf.close()

str_date = '{year}{month}{day}'.format(year='2018', month='08', day='22')  #改这里，日期
parser = argparse.ArgumentParser(description='Get the data info')
#parser.add_argument('-p', '--parent_dir',help='the parent folder of image', default='C:\\Users\\Administrator\\Desktop\\train_data-2018-04-12\\')  #windows系统下用\\
parser.add_argument('-p', '--parent_dir',help='the parent folder of image', default='/home/hyl/data/data/predict_data/predict_data-'+str_date[0:4]+'-'+str_date[4:6]+'-'+str_date[6:8])  #windows系统下用\\  改文件目录
parser.add_argument('-d', '--folder_name',help='the origin folder of image', default='origin')
args = parser.parse_args()


if __name__  == '__main__':
    if args.parent_dir and args.folder_name:
        create_origin(args.parent_dir, args.folder_name)
        new_dir = rot90(args.parent_dir, args.folder_name)
        rename_dir = rename_image(args.parent_dir,'/home/hyl/data/data/predict_data/predict_data-'+str_date[0:4]+'-'+str_date[4:6]+'-'+str_date[6:8]+'/origin_with_rot90') #改文件目录
        # 做完上面三步就跑open_image.py
        #create_zip(args.parent_dir + '\\origin_with_rot90_rename', args.parent_dir)

        # create_zip(args.parent_dir + '\\JPEGImages', args.parent_dir)#做这一步的时候，用#注释掉前面三步


# if __name__  == '__main__':
#     src = 'D:\\2018-04-18-all\\origin_with_rot90_rename'

#     create_zip(src)
    # for i in range(1000):
    #     if i % 200 == 0:
    #         print(i)

"""
run this script,then run the test.py to create Annotations file
"""
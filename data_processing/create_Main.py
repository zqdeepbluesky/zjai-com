#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 3/29/2018 10:53 AM 
# @Author : sunyonghai 
# @File : create_Main.py 
# @Software: ZJ_AI
# =========================================================
import argparse
import random
import os
from data_processing.io_utils import mkdir
from config import ROOT_HOME

def _create_Main(path):
    '''
    create the trainval.txt and test.txt for train.
    trainval data : test data = 5:1
    :param path:
    :return:
    '''
    scale = 10
    image_dir = os.path.join(path, 'JPEGImages')
    anno_dir = os.path.join(path, 'Annotations')
    ImageSets_path = os.path.join(path, 'ImageSets')
    main_dir = os.path.join(ImageSets_path, 'Main')

    mkdir(main_dir)

    imgs = os.listdir(image_dir)
    random.shuffle(imgs)

    trainval_images = []
    test_images = []
    for i in range(len(imgs)//scale, len(imgs)):
        s = imgs[i]
        trainval_images.append(s.split('.')[0] + '\n')

    for i in range(len(imgs)//scale):
        s = imgs[i]
        test_images.append(s.split('.')[0] + '\n')

    with open(main_dir+'/trainval.txt','w+') as f:
        f.writelines(trainval_images)
        print("{}, numbers:{}".format(main_dir + '/trainval.txt', len(trainval_images)))
    with open(main_dir+'/test.txt','w+') as f:
        f.writelines(test_images)
        print("{}, numbers:{}".format(main_dir + '/test.txt', len(test_images)))

    print('total: {}'.format(len(imgs)))
    print('step: {}'.format(len(trainval_images)//2+1))

def create_sub_Main(dirs):
    data_paths = [os.path.join(ROOT_HOME, s) for s in dirs]
    for data_dir in data_paths:
        _create_Main(data_dir)

def create_subs_Main(data_paths):
    # dirs = ['data/train_data-2018-3-7', 'data/train_data-2018-3-16']
    # data_paths = [os.path.join(ROOT_HOME, s) for s in dirs]
    for data_dir in data_paths:
        _create_Main(data_dir)

def create_txt(data_dir):
    _create_Main(data_dir)

def create_txts(data_dirs):
    for data_dir in data_dirs:
        _create_Main(data_dir)

# if __name__ == "__main__":
#     dirs = ['data_52/train_data-2018-04-18']
#     data_paths = [os.path.join(ROOT_HOME, s) for s in dirs]
#     create_subs_Main(data_paths)

parser = argparse.ArgumentParser(description='Get the data info')
parser.add_argument('-d', '--datadir', help='path in server', default='/home/syh/train_data/test')
args = parser.parse_args()

if __name__ == "__main__":
    # data_dir = '/home/syh/disk/train/all_train_data'

    data_dir = args.datadir
    # data_dir = '/home/syh/all_train_data'
    create_txt(data_dir)
#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 3/16/2018 9:51 AM 
# @Author : sunyonghai 
# @File : data_processing_.py
# @Software: ZJ_AI
# =========================================================

import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2

import io_utils

#此函数用来复制annotations对应的jpeg
#合格的在JPEGImages_marked文件夹中，不合格的在JPEGImages中
#对原始文件操作时，最好用复制or移动，不要delete
def pick_marked(input_path):
    #data_paths = [os.path.join(input_path, s) for s in ['all_data']]
    data_path=input_path
    print('Parsing annotation files')

    #for data_path in data_paths:
    annot_path = os.path.join(data_path, 'Annotations')
    imgs_path = os.path.join(data_path, 'JPEGImages')
    imgs_out_path = os.path.join(data_path, 'JPEGImages_marked')
    io_utils.mkdir(imgs_out_path)

    annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
    for annot in annots:
        try:
            et = ET.parse(annot)
            element = et.getroot()
            element_filename = element.find('filename').text
            filepath = os.path.join(imgs_path, element_filename)
            io_utils.move(filepath, imgs_out_path)
        except Exception as e:
            print('Exception in pascal_voc_parser: {}'.format(e))
            continue

def copy_marked(data_path):

    if not os.path.isdir(data_path):
        print('input_path is not a dir: {}'.format(data_path))
        return

    # for data_path in input_path:
    annot_path = os.path.join(data_path, 'Annotations')
    imgs_path = os.path.join(data_path, 'JPEGImages')
    imgs_out_path = os.path.join(data_path, 'JPEGImages_marked')
    io_utils.mkdir(imgs_out_path)

    annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
    for annot in annots:
        try:
            et = ET.parse(annot)
            element = et.getroot()
            element_filename = element.find('filename').text
            filepath = os.path.join(imgs_path, element_filename)
            io_utils.copy(filepath, imgs_out_path)
        except Exception as e:
            print('Exception in pascal_voc_parser: {}'.format(e))
            continue


# def rot90(input_path):
#     data_paths = [os.path.join(input_path, s) for s in ['train_data-2018-3-20']]
#     print('Parsing annotation files')
#
#     for data_path in data_paths:
#         imgs_path = os.path.join(data_path, 'JPEGImages')
#         imgs_out_path = os.path.join(data_path, 'JPEGImages_with_rot90')
#         io_utils.delete_file_folder('JPEGImages_with_rot90')
#         io_utils.mkdir(imgs_out_path)
#
#         images = [os.path.join(imgs_path, s) for s in os.listdir(imgs_path)]
#         for image_file in images:
#             try:
#                 img = cv2.imread(image_file)
#                 width = img.shape[0]
#                 height = img.shape[1]
#                 if width > height:
#                     image = np.array(np.rot90(img, 1))
#                     image = image.copy()
#                 else:
#                     image = img
#                 name = image_file.split('/')[-1]
#                 save_path = os.path.join(imgs_out_path, name)
#                 print(image.shape)
#                 # image = cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0]*0.5)), interpolation=cv2.INTER_CUBIC)
#                 # print(image.shape)
#                 cv2.imwrite(save_path, image)
#                 print(save_path)
#             except Exception as e:
#                 print('Exception in pascal_voc_parser: {}'.format(e))
#                 continue

def remove_error(input_path):
    data_path = os.path.join(input_path, 'train_data-2018-3-7')
    annot_path = os.path.join(data_path, 'Annotations')
    annot_error_path = os.path.join(data_path, 'Annotations_error')
    imgs_path = os.path.join(data_path, 'JPEGImages')
    imgs_error_path = os.path.join(data_path, 'JPEGImages_error')

    imgs_error = os.path.join(data_path, 'error')
    for error_image in os.listdir(imgs_error):
        error_image_name = error_image.split('.')[0]

        for annot_xml in os.listdir(annot_path):
            xml_name = annot_xml.split('.')[0]
            if error_image_name == xml_name:
                filepath = os.path.join(annot_path, annot_xml)
                io_utils.move(filepath, annot_error_path)
                break

        for image in os.listdir(imgs_path):
            image_name = image.split('.')[0]
            if error_image_name == image_name:
                filepath = os.path.join(imgs_path,image)
                io_utils.move(filepath, imgs_error_path)
                break

def rename_image(input_path):
    # data_path = os.path.join(input_path, 'from_internet')
    data_path = input_path
    # imgs_path = os.path.join(data_path, 'wwsp-wznn-hz-yw-125ml')
    # imgs_rename_path = os.path.join(data_path, 'wwsp-wznn-hz-yw-125ml_rename')
    # imgs_path = os.path.join(data_path, 'yl-ylcnn-pz-yw-250ml')
    # imgs_rename_path = os.path.join(data_path, 'yl-ylcnn-pz-yw-250ml_rename')
    data_paths = [os.path.join(data_path, s) for s in ['all_data']]
    for data_dir in data_paths:
        # prefix = data_dir.split('-')[1]
        prefix = 'train'
        imgs_rename_path = '{}_rename'.format(data_dir)
        io_utils.delete_file_folder(imgs_rename_path)
        io_utils.mkdir(imgs_rename_path)

        idx = 1000
        for s in os.listdir(data_dir):
            old = os.path.join(data_dir, s)
            new = os.path.join(imgs_rename_path, '{}_20180319_{}.jpg'.format(prefix, idx))
            io_utils.rename(old,new)
            idx = idx+1

def rename_suffix_image(input_path):
    data_dir = os.path.join(input_path, 'all_data/JPEGImages')
    imgs_rename_path = '{}_rename'.format(data_dir)
    io_utils.delete_file_folder(imgs_rename_path)
    io_utils.mkdir(imgs_rename_path)
    for s in os.listdir(data_dir):
        old = os.path.join(data_dir, s)
        name = s.split('.')[0]
        new = os.path.join(imgs_rename_path, '{}.jpg'.format(name))
        io_utils.rename(old,new)

def rename_suffix2_image(input_path, old_suffix='JPG',new_suffix='jpg'):
    '''
    JPG ->jpg
    '''
    data_dir = os.path.join(input_path, 'JPEGImages')
    ls = os.listdir(data_dir)
    idx = 0
    for i in ls:
        sub_path = os.path.join(data_dir, i)
        if os.path.isdir(sub_path):
            rename_suffix2_image(sub_path, old_suffix, old_suffix)
        else:
            file_post = str(i.split('.')[-1])
            if file_post == old_suffix:
                os.rename(sub_path, str(sub_path.split('.')[0]) + '.' + new_suffix)
                print('[{index}]:找到文件{srcnam},已修改成:{dicname}'.format(index=idx, srcnam=sub_path, dicname=str(i.split('.')[0]) + '.' + new_suffix))
                idx+=1

def get_xml_by_imagename():
    image_dir = 'data/train_data-2018-3-29/JPEGImages'
    annotations_tar_dir = 'data/train_data-2018-3-29//Annotations'
    annotations_src_dir = 'data/train_data-2018-3-29/Annotations'
    for image_file in os.listdir(image_dir):
        image_name = image_file.split('.')[0]

        for annot_xml in os.listdir(annotations_src_dir):
            xml_name = annot_xml.split('.')[0]
            if image_name == xml_name:
                filepath = os.path.join(annotations_src_dir, annot_xml)
                io_utils.move(filepath, annotations_tar_dir)
                break

def trainval(input_path):
    data_dir =  os.path.join(input_path, 'JPEGImages')
    ImageSets_dir =  os.path.join(input_path, 'ImageSets/Main')

    test_file = os.path.join(ImageSets_dir,'test.txt')
    val_file = os.path.join(ImageSets_dir,'val.txt')
    train_file = os.path.join(ImageSets_dir, 'val.txt')
    trainval_file = os.path.join(ImageSets_dir, 'trainval.txt')
    with open(trainval_file, 'w+') as f:
        image_names = []
        for s in os.listdir(data_dir):
            image_names.append(s.split('.')[0] + '\n')
            # print(s)
        print(len(image_names))
        f.writelines(image_names)
# if __name__ == '__main__':
#     input_path = 'data/'
#     pick_marked(input_path)

# if __name__ =='__main__':
#     input_path = 'data/'
#     rot90(input_path)

# if __name__ == '__main__':
#     input_path = 'data/'
#     remove_error(input_path)


# if __name__ == '__main__':
#     input_path = 'data/'
#     rename_image(input_path)

if __name__ == '__main__':
    #get_xml_by_imagename()
    pick_marked("D:\\all_data\\predict_data-2018-05-16") #修改路径，日期！

# if __name__ == '__main__':
#     input_path = 'data/all_data/'
#     trainval(input_path)

# if __name__ == '__main__':
#     input_path = 'data'
#     rename_suffix_image(input_path)

# if __name__ == '__main__':
#     input_path = 'data/train_data-2018-3-20_2'
#     # data_paths = [os.path.join(input_path, s) for s in ['train_data-2018-3-20']]
#
#     copy_marked(input_path)

# if __name__ == '__main__':
#     input_path = 'data/train_data-2018-3-7'
#     # data_paths = [os.path.join(input_path, s) for s in ['train_data-2018-3-20']]
#
#     rename_suffix2_image(input_path, old_suffix='JPG', new_suffix='jpg')
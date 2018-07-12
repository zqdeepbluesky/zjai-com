#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 3/29/2018 11:19 AM 
# @Author : sunyonghai 
# @File : rename_image.py 
# @Software: ZJ_AI
# =========================================================
import argparse
import os
import datetime
from data_processing.io_utils import rename
from data_processing.xml_utils import read_xml, write_xml

def rename_image_file(src, dst):
    """
    Modify the image file name.
    :param src: path of source image
    :param dst: path of target image
    :return: True or False
    """
    return rename(src, dst)

def rename_annotation_file(src, dst):
    """
    Modify the annotation xml file name
    :param src: path of source annotation xml-file
    :param dst: path of target annotation xml-file
    :return: True or False
    """
    return rename(src, dst)

def rename_image_in_xml(annot_path, name):
    """
    Modify the image name of annotation xml-file.
    :param annot_path: the path of annotation xml-file.
    :param name: the new name of image
    :return: True or False
    """
    if annot_path=='' or name =='':
        print('Invalid parameter')
        return False
    try:
        et = read_xml(annot_path)
        element = et.getroot()
        element_filename = element.find('filename').text
        node = element.find('filename')
        node.text = name
        write_xml(et, annot_path)
        print("Change the name from {} to {}.".format(element_filename, name))
        return True
    except Exception as ex:
        print(ex)
        return False

parser = argparse.ArgumentParser(description='Get the data info')
parser.add_argument('-p', '--parent_dir',help='the parent folder of image', default='/home/syh/RetinaNet/data/train_data-2018-03-30')
parser.add_argument('-j', '--JPEGImages',help='the folder of image', default='JPEGImages')
parser.add_argument('-a', '--Annotations',help='the folder of annotation', default='Annotations')
args = parser.parse_args()

if __name__ == '__main__':
    if args.parent_dir != '' or args.JPEGImages != '' or args.Annotations != '':

        JPEGImages_dir = os.path.join(args.parent_dir, args.JPEGImages)
        Annotations_dir = os.path.join(args.parent_dir, args.Annotations)
        prefix = 'train'
        idx = 1000
        cur_date = datetime.datetime.now()
        str_date = '{year}{month}{day}'.format(year=cur_date.year, month=cur_date.month, day=cur_date.day)
        # str_date = '{year}{month}{day}'.format(year='2018', month='03', day='16')
        for s in os.listdir(JPEGImages_dir):
            old_path = os.path.join(JPEGImages_dir, s)
            old_name = s.split('.')[0]

            new_name = '{}_{}_{}'.format(prefix, str_date, idx)

            new_path = os.path.join(JPEGImages_dir, new_name+'.jpg')
            rename_image_file(old_path, new_path)

            old_annot_path = os.path.join(Annotations_dir, old_name+'.xml')
            new_annot_path = os.path.join(Annotations_dir, new_name+'.xml')
            # rename_image_in_xml(old_annot_path, new_name+'.jpg')

            # rename_annotation_file(old_annot_path, new_annot_path)
            idx = idx + 1
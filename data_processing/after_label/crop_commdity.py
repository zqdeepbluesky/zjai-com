# -*- coding: utf-8 -*-
# @Time    : 5/25/2018 10:48 AM
# @Author  : sunyonghai
# @File    : crop_commdity.py
#multiprocessing "cut_bbox"
# @Software: ZJ_AI
import argparse
import json
import os
from multiprocessing import Process, Pool,cpu_count

import PIL.Image
import cv2
import xml.etree.ElementTree as ET

import utils.io_utils


def get_data(data_path):
    '''
    读取pascal voc 数据
    1)基本信息设置
    2)读取XML文件
    3)是否显示图片
    :param input_path: 只需要给定到VOC所在的文件夹，不需要知道给定到具体的版本
    如：input_path = 'F:/study_files/faster_rcnn/training_data/VOCdevkit'
    :return:
    '''
    '''
    图片的高度，宽度，路径，和所处训练集和框。
    其中bboxes: 其是一个list,每一条信息是以字典形式存储包含了一个box的所有信息。
    有难度，类别，上下两点的坐标。下面是一个示列:
    [{'height': 500, 
    'imageset': 'trainval',
    'width': 486, 
    'filepath':'data/VOC2012/JPEGImages/2007_000027.jpg',
    'bboxes': [
                {
                    'x1': 174，
                    'x2': 349, 
                    'y1': 101, 
                    'y2': 351,
                    'class': 'person', 
                    'difficult':False,
                    }，
                    {
                    'x1': 174，
                    'x2': 349, 
                    'y1': 101, 
                    'y2': 351,
                    'class': 'person', 
                    'difficult':False,
                }
               ]
    }]
    '''
    all_imgs = []  # 其是一个list,每一条信息是以字典形式存储包含了一张图片的所有信息。

    classes_count = {}  # classes_count:是一个字典，其存储类别和其对应的总个数。比如：{'person': 12, 'horse': 21}

    class_mapping = {}  # 是一个字典，其对应每一个类别对应的编号，one-hot {'person': 0, 'horse': 1}

    print('Parsing annotation files')
    #
    # for data_path in data_paths:
    annot_path = os.path.join(data_path, 'Annotations')
    imgs_path = os.path.join(data_path, 'JPEGImages')

    annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
    idx = 0
    for annot in annots:
        try:
            idx += 1

            et = ET.parse(annot)
            element = et.getroot()  # 得到xml的根

            element_objs = element.findall('object')
            element_filename = element.find('filename').text
            element_width = int(element.find('size').find('width').text)
            element_height = int(element.find('size').find('height').text)

            if len(element_objs) > 0:
                # annotation format 封装后的注释格式
                annotation_data = {'filepath': os.path.join(imgs_path, element_filename),
                                   'image_name': element_filename,
                                   'width': element_width,
                                   'height': element_height,
                                   'bboxes': []}

            for element_obj in element_objs:
                ## 直接目标检测
                class_name = element_obj.find('name').text

                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

                if class_name not in class_mapping:
                    class_mapping[class_name] = len(class_mapping)

                obj_bbox = element_obj.find('bndbox')
                x1 = int(round(float(obj_bbox.find('xmin').text)))
                y1 = int(round(float(obj_bbox.find('ymin').text)))
                x2 = int(round(float(obj_bbox.find('xmax').text)))
                y2 = int(round(float(obj_bbox.find('ymax').text)))
                difficulty = int(element_obj.find('difficult').text) == 1
                # annotation format of bounding box 矩形框的封装格式
                annotation_data['bboxes'].append(
                    {'class': class_name,
                     'x1': x1,
                     'x2': x2,
                     'y1': y1,
                     'y2': y2,
                     'difficult': difficulty})
            all_imgs.append(annotation_data)
        except Exception as e:
            print('Exception in pascal_voc_parser: {}'.format(e))
            continue
    return all_imgs

def crop_bbox(src_img_path, box, save_path):
    try:
        img = PIL.Image.open(src_img_path)
        bbox = img.crop(box)
        bbox.save(save_path)
        print("save: {}".format(save_path))
    except Exception as ex:
        print(ex)

def process(data):
    src_img_path = data['filepath']
    bboxes = data['bboxes']
    class_count={}
    for bbox in bboxes:
        class_label = bbox['class']
        if class_label not in class_count:
            class_count[class_label] = 0
        else:
            class_count[class_label] += 1

        box_xy = (bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'])

        # if class_count[class_label] == 0:
        #     bbox_name ="{}{}".format(data['image_name'].split('.')[0], '.jpg')
        # else:
        #     bbox_name ="{}-{}{}".format(data['image_name'].split('.')[0], class_count[class_label], '.jpg')

        bbox_name ="{}-{}{}".format(data['image_name'].split('.')[0], class_count[class_label], '.jpg')

        class_label_folder = os.path.join(save_path, class_label)
        io_utils.mkdir(class_label_folder)
        save_bbox_path = os.path.join(class_label_folder, bbox_name)

        crop_bbox(src_img_path, box_xy,  save_bbox_path)

parser = argparse.ArgumentParser(description='Get the data info')
parser.add_argument('-i', '--input',help='directory of data path', default= 'test_folder')
parser.add_argument('-o', '--output',help='output diretory', default='')
args = parser.parse_args()

# data_path = args.input
# if args.output == '':
#     save_path = os.path.join(args.input, 'crop_commdity')
# else:
#     save_path = args.output

def main(data_path, save_path):
    all_data = get_data(data_path)
    cpus = cpu_count() // 2 #取整除
    p = Pool(cpus)
    p.map_async(process, all_data)

    p.close()
    p.join()

if __name__ == '__main__':

    data_path = args.input
    if args.output=='':
        save_path = os.path.join(args.input, 'crop_commdity')
    else:
        save_path = os.path.join(args.output, 'crop_commdity')
    io_utils.mkdir(save_path)

    main(data_path, save_path)

    print('finished')

"""
cd /home/syh/RetinaNet/data_processing

"""
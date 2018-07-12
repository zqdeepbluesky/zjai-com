#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 3/30/2018 2:01 PM 
# @Author : sunyonghai 
# @File : voc_parser.py 
# @Software: ZJ_AI
#此程序是处理标注好的结果，包括统计商品种类及个数
# =========================================================
import json
import os
import pprint

import cv2
import xml.etree.ElementTree as ET
import numpy as np


def get_data(input_path):
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

    visualise = False  #是否显示图片

    # train VOC2012
    # data_paths = [os.path.join(input_path,s) for s in ['VOC2012']]
    # data_paths = [os.path.join(input_path, s) for s in ['VOC2007', 'VOC2012']]
    # 如果有新的数据集，只需要添加到列表即可

    # train commdity
    # data_paths = [os.path.join(input_path, s) for s in ['train_data-2018-3-7','train_data-2018-3-16','train_data-2018-3-21']]
    # data_paths = [os.path.join(input_path, s) for s in ['train_data-2018-3-20_2']]
    #data_paths = [os.path.join(input_path, s) for s in ['all_data']]
    data_path = input_path
    # data_paths = [os.path.join(input_path, s) for s in ['train_data-2018-3-29']]
    # data_paths = [os.path.join(input_path, s) for s in ['train_data-2018-3-7']]

    print('Parsing annotation files')

    #for data_path in data_paths:

    annot_path = data_path
    imgs_path = os.path.join(data_path, 'JPEGImages')
    '''
    imgsets_path_trainval = os.path.join(data_path, 'ImageSets\\Main\\trainval.txt')
    imgsets_path_test = os.path.join(data_path, 'ImageSets\\Main\\test.txt')

    # 得到训练与测试集图片文件的名称，这是为以后判断图片是属于哪个集而准备的
    # 由于没有测试，和交叉检验，所以这个暂时没起到作用
    trainval_files = []
    test_files = []
    try:
        with open(imgsets_path_trainval) as f:
            for line in f:
                trainval_files.append(line.strip() + '.jpg')
    except Exception as e:
        print(e)

    try:
        with open(imgsets_path_test) as f:
            for line in f:
                test_files.append(line.strip() + '.jpg')
    except Exception as e:
        if data_path[-7:] == 'VOC2012':
            # this is expected, most pascal voc distibutions dont have the test.txt file
            pass
        else:
            print(e)
    '''
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
                                   'width': element_width,
                                   'height': element_height,
                                   'bboxes': []}
                '''
                if element_filename in trainval_files:
                    annotation_data['imageset'] = 'trainval'
                elif element_filename in test_files:
                    annotation_data['imageset'] = 'test'
                else:
                    annotation_data['imageset'] = 'trainval'
               '''
            for element_obj in element_objs:

                ## 做分类用的
                # class_label = element_obj.find('name').text
                # if class_label != 'other':
                #     class_name = class_label.split('-')[2]
                # else:
                #     class_name = class_label # 'other'
                #     continue # 不要other类框

                ## 直接目标检测
                class_name = element_obj.find('name').text

                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

                if class_name not in class_mapping:
                    class_mapping[class_name] = len(class_mapping)

                obj_bbox = element_obj.find('bndbox')
                x1 = int(round(float(obj_bbox.find('xmin').text)))  #round返回浮点数的四舍五入值
                y1 = int(round(float(obj_bbox.find('ymin').text)))
                x2 = int(round(float(obj_bbox.find('xmax').text)))
                y2 = int(round(float(obj_bbox.find('ymax').text)))
                difficulty = int(element_obj.find('difficult').text) == 1 #先判断是否等于1
                # annotation format of bounding box 矩形框的封装格式
                annotation_data['bboxes'].append(
                    {'class': class_name,
                     'x1': x1,
                     'x2': x2,
                     'y1': y1,
                     'y2': y2,
                     'difficult': difficulty})
            all_imgs.append(annotation_data)

            if visualise:  #会一张一张地显示标定框
                img = cv2.imread(annotation_data['filepath'])
                for bbox in annotation_data['bboxes']:
                    cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 0, 255))
                cv2.imshow('img', img)
                cv2.waitKey(0)

        except Exception as e:
            print('Exception in pascal_voc_parser: {}'.format(e))
            continue
    return all_imgs, classes_count, class_mapping

def sort_by_value(d): #no use here
    items=d.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort()
    return [backitems[i][1] for i in range(0, len(backitems))]

def save_mapping(class_mapping):
    # with open('/home/syh/RetinaNet/mapping.json', 'w+') as f:
    with open('D:\\2018-04-19\\2018-04-22-Annotations\\mapping.json', 'w+') as f:
        json.dump(class_mapping, f,sort_keys=True) # json.dump()用于将dict类型的数据转成str，并写入到json文件中
        print('save the class mapping to D:\\2018-04-22\\2018-04-22-Annotations\\mapping.json')

if __name__ == '__main__':
    input_path = 'D:\\all_data\\predict_data-2018-05-16\\Annotations' #修改路径，日期！
    all_imgs, classes_count, class_mapping = get_data(input_path)
    #save_mapping(class_mapping)

    print("class_mapping:----------------------------------")
    pprint.pprint(sorted(class_mapping.items(), key=lambda item:item[1]))

    print("classes_count:----------------------------------")
    # sorted(classes_count.items(), key=lambda item: item[1])
    # for key in sorted(classes_count.items(), key=lambda item: item[1]).keys():
    #     print(key)
    a = sorted(classes_count.items(), key=lambda item: item[0])
    for idx in range(len(a)):
        print(a[idx][0]+":"+str(a[idx][1]))

    print("classes:----------------------------------")
    # sorted(classes_count.items(), key=lambda item: item[1])
    # for key in sorted(classes_count.items(), key=lambda item: item[1]).keys():
    #     print(key)
    a = sorted(classes_count.items(), key=lambda item: item[0])
    for idx in range(len(a)):
        print(a[idx][0])
    # pprint.pprint(sorted(classes_count.items(), key=lambda item: item[0]))

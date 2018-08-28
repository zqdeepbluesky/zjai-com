# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 8/23/2018 3:23 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import random
import numpy as np

import data_processing.fusion.fusion_utils
import data_processing.fusion.generator
import data_processing.fusion.iou_utils


# 1. Load data
# 2. fusion operation
# 3. save data

threshold = 0.001
scale = 6

class ImageInfo():
    pass

def randrom_xy(max_xy):
    x = random.randint(0, max_xy[0])
    y = random.randint(0, max_xy[1])
    return x,y


def check_iou(boxes, box):
    for b in boxes:
        res = data_processing.fusion.iou_utils.calc_iou(b, box)
        if res > threshold:
            return False
    else:
        return True

def check_edge(box, max_xy):
    if box[2] >= max_xy[0]-1 or box[3] >= max_xy[1]-1:
        return False
    return True

def create_image_name():
    cur_date = datetime.datetime.now()
    str_date = '{year}-{month}-{day}'.format(year=cur_date.year, month=cur_date.month, day=cur_date.day)
    name  = "{}_{}".format('fusion', str_date)
    return name

# get bg
# get object
def do_fusion(num, idx=10000):
    #get buffer
    buffer_size = (3600, 3600)

    # get bg
    # get object

    classes = data_processing.fusion.generator.read_classes(classes_path)
    data_gen = data_processing.fusion.generator.Generator(data_path, bg_path, classes)

    while(idx < 10000 + num ):
        objs, labels, bg = data_gen.next()
        bg_buffer = np.zeros((buffer_size + (4,)), np.uint8)

        boxes = []
        label_names = []
        timeout = 1
        for (obj , label) in zip(objs, labels) :
            while(timeout < 100):
                # create a random (x,y)
                x,y = randrom_xy(buffer_size)
                box = [x,y, x+obj.shape[0], y+obj.shape[1]]
                timeout = timeout + 1
                # print(box)
                if  check_edge(box, buffer_size) and check_iou(boxes, box):
                    # calc iou
                    # paste
                    bg_buffer = data_processing.fusion.fusion_utils.paste_obj(bg_buffer, obj, box)

                    label_names.append(data_gen.label_to_name(np.argwhere(label==1)[0][0]))
                    boxes.append(box)
                    break

        # composite

        bg_im = data_processing.fusion.fusion_utils.resize_image(bg[:,:,::-1], (buffer_size[1],buffer_size[0]))
        fusion_img = data_processing.fusion.fusion_utils.composite_bg(bg_im, bg_buffer)

        im_name = create_image_name()+"_"+str(idx)
        im_info = ImageInfo()
        im_info.width = buffer_size[0]
        im_info.height = buffer_size[1]
        im_info.path = ''
        im_info.name = im_name
        im_info.image_extension = 'jpg'
        im_info.channel = 3

        # save image
        image_output_path = os.path.join(train_data_path, 'JPEGImages', im_info.name + '.' +im_info.image_extension)
        fusion_img  = data_processing.fusion.fusion_utils.resize_image(fusion_img, (buffer_size[1]//scale, buffer_size[0]//scale))

        data_processing.fusion.fusion_utils.save_image(image_output_path, np.transpose(fusion_img,[1,0,2]))

        boxes = np.array(boxes)[:6]//scale
        # save xml
        image_xml_dir = os.path.join(train_data_path, 'Annotations')
        data_processing.fusion.fusion_utils.save_annotations(image_xml_dir, im_info, boxes, label_names)

        if idx % 100 == 0:
            print(idx)
        idx = idx+1

if __name__ == '__main__':
    root_home = '/home/syh/tf-faster-rcnn/data/fusion/'
    data_path = os.path.join(root_home, 'mask')
    bg_path = os.path.join(root_home, 'bg/background_min_2000/')
    classes_path = os.path.join(root_home,'/home/syh/tf-faster-rcnn/data_processing/fusion/mapping_all.json')
    name = create_image_name()
    train_data_path = os.path.join(root_home, 'output/{}_20000/'.format(name))

    do_fusion(num = 20000, idx=10000)
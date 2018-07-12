# -*- coding: utf-8 -*-
# @Time    : 5/17/2018
# @Author  : CarrieChen
# @File    : cut_bbox.py
# @Software: ZJ_AI
# this code is for cut out bbox images and put the same kind of labels into a folder.


import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

import io_utils
from io_utils import *
from PIL import Image
import matplotlib.pyplot as plt

def dict_init(input_path):
    annot_path = input_path+"\\Annotations"
    count = {}
    annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
    list=[]
    img_name_list=[]
    for annot in annots: #read a xml
        try:
            et = ET.parse(annot)
            element = et.getroot()
            element_objs = element.findall('object')
            img_name = element.find('filename').text
            for element_obj in element_objs:
                class_name = element_obj.find('name').text  #find label
                if class_name in list:  #judge is important
                    if img_name in img_name_list:
                        count[class_name][img_name] = count[class_name][img_name] + 1
                    else:
                        count[class_name][img_name]=0
                else:
                    count[class_name]={}
                    count[class_name][img_name] = 0
                    list.append(class_name)

        except Exception as e:
            print('Exception in pascal_voc_parser: {}'.format(e))
            continue
    return count

def count_and_cut(input_path,count):
    annot_path = input_path + "\\Annotations"
    img_path = input_path + "\\JPEGImages"
    save_folder = input_path + "\\Cutout"
    io_utils.mkdir(save_folder)
    io_utils.remove_all(save_folder)
    annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
    for annot in annots:  # read a xml
        try:
            et = ET.parse(annot)
            element = et.getroot()
            element_objs = element.findall('object')
            img_name = element.find('filename').text

            new_img_path = img_path + "\\" + img_name  # find uncut image
            for element_obj in element_objs:
                class_name = element_obj.find('name').text  # find label
                count[class_name][img_name] = count[class_name][img_name] + 1
                save_path = save_folder + "\\" + class_name
                save_name = img_name.split('.')[0] + '-' + str(count[class_name][img_name]) + '.jpg'
                io_utils.mkdir(save_path)
                xmin = int(element_obj.find("bndbox").find("xmin").text)  # find bbox boundary
                ymin = int(element_obj.find("bndbox").find("ymin").text)
                xmax = int(element_obj.find("bndbox").find("xmax").text)
                ymax = int(element_obj.find("bndbox").find("ymax").text)
                box = (xmin, ymin, xmax, ymax)
                img = Image.open(new_img_path)
                region = img.crop(box)
                region.save(save_path + "\\" + save_name)
        except Exception as e:
            print('Exception: {}'.format(e))
            continue


input_path='D:\\all_data\\predict_data-2018-05-15'  #change there!!!

if __name__=="__main__":
    null_dict=dict_init(input_path)
    count_and_cut(input_path,null_dict)
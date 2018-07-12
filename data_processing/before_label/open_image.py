# -*- coding: utf-8 -*-
# @Time    : 5/10/2018 11:53 AM
# @Author  : sunyonghai,carriechen
# @File    : open_image.py
# @Software: ZJ_AI
#this code is for open images and test whether there are complete

import numpy as np
import time
from PIL import Image
import os
import cv2

#read by PIL
def read_image_bgr_PIL(path):
    try:
        image = np.asarray(Image.open(path).convert('RGB'))
        if image is None:
            raise Exception("Invalid image!", path)
    except Exception as ex:
        print(path)
        print(ex)
    return image[:, :, ::-1].copy()

#read by opencv
def read_image_bgr_opencv(path):
    try:
        image=np.asarray(cv2.imread(path,cv2.IMREAD_COLOR))
        if image is None:
            raise Exception("Invalid image!", path)
    except Exception as ex:
        print(path)
        print(ex)
    #return image[:, :, ::-1].copy()

if __name__ == '__main__':
    # data_path = '/home/syh/all_train_data/JPEGImages'
    #data_path = 'D:\\all_data\\predict_data-2018-05-21\\JPEGImages' #修改这里的路径
    data_path="D:\\标签文档2018-05-29\\标签文档2018-05-29\\食品图+标签166类\\食品图+标签166类"

    for files in os.listdir(data_path):
        path = os.path.join(data_path,files)
        #st = time.time()
        read_image_bgr_PIL(path)
        read_image_bgr_opencv(path)
        #end = time.time()

        #str_info = "{}: read time:{} ms".format(files, str(1000 * (end - st)))
        #print(str_info)
    print("no invalid image")
#
# if __name__ == '__main__':
#     data_path = '/home/syh/all_train_data/JPEGImages'
#
#     count = 0
#     for _ in range(0, 100):
#         path = '/home/syh/all_train_data/JPEGImages/train_20180409_1654.jpg'
#         st = time.time()
#         read_image_bgr(path)
#         end = time.time()
#         str_info = "read time:{} ms".format(str(1000 * (end - st)))
#         print(str_info)
#
#         count += 1000 * (end - st)
#
#     print("total:{}".format(count // 100))
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 4/23/2018 17:26 AM
# @Author : CarrieChen
# @File : batch_handle_pic.py
# @Software: ZJ_AI
#this code is for batch handle picture.such as batch resize background
# =========================================================

from PIL import Image
import os.path
import glob
import io_utils
import cv2

#resize and rename
def handle(img, min_side, max_side,output_path,n):
    img=cv2.imread(img)
    (rows, cols, _) = img.shape
    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)
    output_path=os.path.join(output_path,"background"+"_"+str_date[0:4]+"-"+str_date[4:6]+"-"+str_date[6:8]+"-"+str(n)+".jpg")
    cv2.imwrite(output_path,img) #output img name,change here!!
    #return img, scale



#some path
parent_path="test_folder/Background"
origin_path=os.path.join(parent_path,"origin")
handle_path=os.path.join(parent_path,"after")



str_date = '{year}{month}{day}'.format(year='2018', month='05', day='29')  #改这里，日期

if __name__  == '__main__':
    ncount=10000
    io_utils.mkdir(handle_path)
    io_utils.remove_all(handle_path)
    for f in os.listdir(origin_path):
        img_path=os.path.join(origin_path,f)
        handle(img_path,800,1333,handle_path,ncount)  #background:800*1333
        ncount+=1


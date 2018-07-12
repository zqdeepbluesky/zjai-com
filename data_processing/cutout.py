#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 4/12/2018 9:35 AM 
# @Author : sunyonghai 
# @File : cutout.py 
# @Software: ZJ_AI
# =========================================================

import cv2
import  numpy as np
import os
from config import ROOT_HOME

cutout_dir = os.path.join(ROOT_HOME, 'cutout/')
background_dir = os.path.join(ROOT_HOME, 'background/')

image_path = os.path.join(cutout_dir, '1.jpg')
bg_image_path = os.path.join(background_dir, '050051.jpg')

# # 加载&缩放
# img=cv2.imread(image_path)
#
# img_back=cv2.imread(bg_image_path)
# #日常缩放
# rows,cols,channels = img_back.shape
# img_back=cv2.resize(img_back,None,fx=0.7,fy=0.7)
# #
#
# rows,cols,channels = img.shape
# img=cv2.resize(img,None,fx=0.4,fy=0.4)
#
#
# rows,cols,channels = img.shape#rows，cols最后一定要是前景图片的，后面遍历图片需要用到
#
#
# # 获取背景区域
# #转换hsv
# hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#
# #获取mask
# lower_blue=np.array([78,43,46])
# upper_blue=np.array([110,255,255])
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
#
# cv2.imwrite(cutout_dir+'/mask.jpg', mask)
#
# # mask优化
# #腐蚀膨胀
# erode=cv2.erode(mask,None,iterations=1)
# cv2.imwrite(cutout_dir+'/erode.jpg', erode)
#
# dilate=cv2.dilate(erode,None,iterations=1)
# cv2.imwrite(cutout_dir+'/dilate.jpg', dilate)
#
#
# #替换背景图片
# #遍历替换
# center=[50,50]#在新背景图片中的位置
# for i in range(rows):
#     for j in range(cols):
#         if dilate[i,j]==0:#0代表黑色的点
#             img_back[center[0]+i,center[1]+j]=img[i,j]#此处替换颜色，为BGR通道
# cv2.imwrite(cutout_dir+'/matting.jpg', img_back)


import cv2
import  numpy as np

img=cv2.imread(image_path)
img_back=cv2.imread(bg_image_path)
#日常缩放
rows,cols,channels = img_back.shape
img_back=cv2.resize(img_back,None,fx=0.7,fy=0.7)
# cv2.imshow('img_back',img_back)

rows,cols,channels = img.shape
img=cv2.resize(img,None,fx=0.4,fy=0.4)
# cv2.imshow('img',img)
rows,cols,channels = img.shape#rows，cols最后一定要是前景图片的，后面遍历图片需要用到

#转换hsv
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#获取mask
# lower_blue=np.array([78,43,46])
# upper_blue=np.array([110,255,255])
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
# cv2.imshow('Mask', mask)

lower_green = np.array([21,120,83])
upper_green =np.array([110,255,255])
mask = cv2.inRange(hsv, lower_green, upper_green)

#腐蚀膨胀
erode=cv2.erode(mask,None,iterations=1)
# cv2.imshow('erode',erode)
dilate=cv2.dilate(erode,None,iterations=1)
# cv2.imshow('dilate',dilate)

#遍历替换
center=[50,50]#在新背景图片中的位置
for i in range(rows):
    for j in range(cols):
        if dilate[i,j]==0:#0代表黑色的点
            img_back[center[0]+i,center[1]+j]=img[i,j]#此处替换颜色，为BGR通道
# cv2.imshow('res',img_back)
cv2.imwrite(cutout_dir+'/matting.jpg', img_back)
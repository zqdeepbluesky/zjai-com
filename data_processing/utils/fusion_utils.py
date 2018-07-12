# -*- coding: utf-8 -*-
# @Time    : 5/28/2018
# @Author  : CarrieChen
# @File    : fusion_utils.py
# @Software: ZJ_AI
# this code is for a series of fusion functions,such as resize,rotate.

from PIL import Image
import os
import random
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from PIL import ImageEnhance
import numpy as np
import matplotlib.pyplot as plt


#scale image according to the rate
def scale(im,rate):
    im_width, im_height = im.size
    im_resized = im.resize((int(im_width*rate),int(im_height*rate)))
    #print("scale success")
    return im_resized

#the angle is anticlockwise
def rotate(im,angle):
    '''
    # 转换为有alpha层
    im2 = im.convert('RGBA')
    # 旋转图像
    rot = im2.rotate(angle,expand=True)
    #与旋转图像大小相同的白色图像
    fff=Image.new('RGBA',rot.size,(255,255,255,1))
    fff.show()
   # fff = Image.new('RGBA', rot.size, (255,) * 4)
    #使用alpha层的rot作为掩码创建一个复合图像
    out=Image.composite(rot,fff,rot)
    out.show()
    #保存你的工作回到mode = '1'或任何..）
    out.convert(im.mode)
    '''
    im_rotate = im.rotate(angle,expand=True) #expand=True means that the size will change
    #print("rotate success")
    return im_rotate

def transpose(im,mode):
    try:
        if mode==['H']:  #horizontal
            im_transpose=im.transpose(Image.FLIP_LEFT_RIGHT)
        elif mode==['V']: #vertical
            im_transpose = im.transpose(Image.FLIP_TOP_BOTTOM)
        #print("transpose success")
        return im_transpose
    except:
        print("invalid transpose mode")
        return False

#if brightness==1,no change;<1,more dark;>1,more bright
def enhance_bright(im,brightness_rate):
    im_enhance_bright = ImageEnhance.Brightness(im)
    image_brightened = im_enhance_bright.enhance(brightness_rate)
    #print("enhance bright success")
    return  image_brightened

#if color_rate<1,lose color;>1,colorful,let red more red and let green more green
def enhance_color(im,color_rate):
    im_enhance_color = ImageEnhance.Color(im)
    image_colored = im_enhance_color.enhance(color_rate)
    #print("enhance color success")
    return image_colored

#if constrast_rate<1,more gray;>1,more constrast
def enhance_constrast(im,contrast_rate):
    enh_con = ImageEnhance.Contrast(im)
    image_contrasted = enh_con.enhance(contrast_rate)
    #print("enhance constrast success")
    return image_contrasted

#enhance sharpness
def enhance_sharpness(im,sharpness_rate):
    im_enhance_sharpness = ImageEnhance.Sharpness(im)
    image_sharped = im_enhance_sharpness.enhance(sharpness_rate)
    #print("enhance sharpness success")
    return image_sharped

#adjust image size to background
def resize(im,part):
    im_width,im_height=im.size
    while (im_width + 50 >= part[0] or im_height + 50 >= part[1]):
        im_width,im_height=im.size
        new_width = int(im_width - 30)  #cut 30 every step
        new_height = int(im_height * new_width / im_width)  # keep the length-width ratio;
        im = im.resize((new_width, new_height), Image.ANTIALIAS)
        im_width, im_height = im.size
    #print("resize success")
    return im

#for test
'''
im=Image.open("test_folder/Src/asm-asmnc-pz-yw-500ml/train_2018-05-08_1003-1.jpg")
#im.show()
#if rotate(im,30):
im=rotate(im,360)  #change here
im.show()
'''


#else:
 #   print('failed')

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 4/23/2018 17:26 AM
# @Author : CarrieChen
# @File : find_empty_file.py
# @Software: ZJ_AI
#此程序是拿来判断图片是否为空/损坏，并把
#输入：图片
#输出：损坏的图片
# =========================================================

import io
import imghdr
from os import PathLike
from PIL import Image
import os
import io_utils

def IsValidImage(dir): #find empty file in a dir
    bValid = True
    num=0
    for s in os.listdir(dir):
        fn=os.path.join(dir,s)
        size=os.path.getsize(fn)
        if size==0:
            print(fn)
            #num=num+1
            os.remove(fn)

    #print(num)

def copyfile(src_dir,com_dir,des_dir):
    #find those files in src_dir but not in com_dir,and copy them to des_dir
    for f in os.listdir(src_dir):
        name=f.split("\\",4)[-1]
        pic=os.path.join(com_dir,name)
        if os.path.exists(pic):
            pass
        else:
            io_utils.copy(os.path.join(src_dir,name),des_dir)
       

if __name__ =='__main__':
    IsValidImage('D:\\all_data\\2018-04-18\\train_data-2018-04-18\\JPEGImages')
    #copyfile('D:\\all_data\\2018-05-04\\origin','D:\\all_data\\2018-05-04\\origin_with_rot90','D:\\all_data\\2018-05-04\\add2')

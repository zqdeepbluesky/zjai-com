#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 4/23/2018 17:26 AM
# @Author : CarrieChen
# @File : many2one.py
# @Software: ZJ_AI
#此程序是拿来把多个子文件夹合成一个文件夹。
# 具体的，收集回来的各小包annotations合成一个。方便检查。
#输入：父文件夹路径
#输出：合成后的文件夹xxxx-xx-xx-Annotations
# =========================================================

import os
import io_utils

def many2one(parent_dir):
    for s in os.listdir(parent_dir):
        childdir=os.path.join(parent_dir,s)
        for t in os.listdir(childdir):
            file=os.path.join(childdir,t)
            io_utils.copy(file,parent_dir)

if __name__  == '__main__':
    many2one("D:\\all_data\\predict_data-2018-05-16\\Annotations")#改路径和日期
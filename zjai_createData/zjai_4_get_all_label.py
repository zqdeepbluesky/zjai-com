#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 7/6/2018 17:26 AM
# @Author : jaykky
# @File : zjai_4_getAllLabel.py
# @Software: ZJ_AI
#此程序是用于统计数据中的label名称列表，便于训练前修改代码中的label。
#属于构造fasterrcnn的数据集的步骤四
#输入：txt文件的父路径，以及文件名称
#输出：所有label的名称列表
# =========================================================

import os
import os.path as osp
import zjai_createData.zjai_3_check_data

def get_all_label(type,root_dir,setname):
    labelList=["__background__"]
    with open(os.path.join(root_dir,"labelCount_{}.txt".format(type)),"r") as f:
        lineList=f.readlines()
        for i in range(len(lineList)-2):
            line=lineList[i].split(":")[0].strip()
            labelList.append(line)
    print(labelList)
    with open(root_dir+"/{}_classes.txt".format(setname),'w') as f:
        f.write("\n".join(labelList))

if __name__=="__main__":
    root_dir = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    fileType = 'trainval'
    root_dir = '/home/hyl/data/ljk/github-pro/zjai-com/data/train_data/train_data-2018-08-15_resize'
    setname="voc"
    zjai_createData.zjai_3_check_data.analysis_data(root_dir,fileType)
    get_all_label(fileType,root_dir,setname)
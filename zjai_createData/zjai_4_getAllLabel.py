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
import sys

def getAlllabel(dataSetDir,type):
    labelList=["__background__"]
    mainDir=osp.join(dataSetDir,"ImageSets","Main")
    with open(os.path.join(mainDir,"labelCount_{}.txt".format(type)),"r") as f:
        lineList=f.readlines()
        for i in range(len(lineList)-2):
            line=lineList[i].split(":")[0].strip()
            labelList.append(line)
    print(labelList)
    with open(dataSetDir+"/classes.txt",'w') as f:
        f.write("\n".join(labelList))

if __name__=="__main__":
    # root_dir = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    root_dir="/home/hyl/data/ljk/project/2-shopDetect/tf-faster-rcnn-master/"
    dataDirs = osp.join(root_dir, 'data', 'train_data')
    fileType = 'trainval'
    getAlllabel(dataDirs,fileType)
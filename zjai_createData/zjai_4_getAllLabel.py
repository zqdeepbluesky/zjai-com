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
import sys
pwd=sys.path[0]
parentDir=os.path.dirname(pwd)
dataSetDir=os.path.join(parentDir,"dataSet")
type='trainval'

def getAlllabel(dataSetDir,type):
    labelList=[]
    with open(os.path.join(dataSetDir,"labelCount_{}.txt".format(type)),"r") as f:
        lineList=f.readlines()
        for i in range(len(lineList)-2):
            line=lineList[i].split(":")[0].strip()
            labelList.append(line)
    print(labelList)

getAlllabel(dataSetDir,type)
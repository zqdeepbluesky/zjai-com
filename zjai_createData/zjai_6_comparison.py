#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 7/6/2018 17:26 AM
# @Author : jaykky
# @File : zjai_6_comparison.py
# @Software: ZJ_AI
#此程序是用于比对模型的识别成果和真实情况，并得到精确率和召回率的模型指标。
#主要方法是对比两个xml文件的object差异
#属于测试fasterrcnn的模型性能
#输入：两个xml文件的路径，
#输出：精确率和召回率
# =========================================================

import xml.etree.ElementTree as ET
import os
from  data_processing.utils import io_utils

def get_xml_label_num(xmlPath):
    '''
    函数用于得到xml文件的object信息
    :param xmlPath:
    :return:
    '''
    if os.path.exists(xmlPath)!=1:
        print(xmlPath)
    et = ET.parse(xmlPath)
    element = et.getroot()
    element_objs = element.findall('object')
    count=len(element_objs)
    labelList=[]
    for element_obj in element_objs:
        node = element_obj.find('name')
        label=node.text
        labelList.append(label)
    return count,labelList

def compare_from_xml(xmlPath1,xmlPath2):
    xmlFileList1=[]
    xmlFileList2 = []
    for xmlFile in os.listdir(xmlPath1):
        xmlFileList1.append(os.path.join(xmlPath1,xmlFile))
        xmlFileList2.append(os.path.join(xmlPath2, xmlFile))

    print(len(xmlFileList1),len(xmlFileList2))
    tp_sum=0
    fp_sum=0
    fn_sum=0
    d_sum=0
    t_sum=0
    for i in range(len(xmlFileList1)):
        tp=0
        fp=0
        fn=0
        xmlFile1=xmlFileList1[i]
        xmlFile2=xmlFileList2[i]
        d_labelNum, d_labelList=get_xml_label_num(xmlFile1)
        t_labelNum, t_labelList=get_xml_label_num(xmlFile2)
        for d_label in d_labelList:
            if d_label in t_labelList:
                labenIndex=t_labelList.index(d_label)
                t_labelList.remove(t_labelList[labenIndex])
                tp+=1
            else:
                fp+=1
            fn=t_labelNum-tp
        tp_sum+=tp
        fp_sum+=fp
        fn_sum+=fn
        # if fp !=0 or fn !=0:
        #     io_utils.copy(xmlFile1.replace("Annotations","JPEGImages").replace(".xml",".jpg"),save_path)
        #     io_utils.copy(xmlFile1,save_path)
        d_sum+=d_labelNum
        t_sum+=t_labelNum
        print(xmlFile1,xmlFile2,tp,fp,fn,d_labelNum,t_labelNum)
    prec=tp_sum/(fp_sum+tp_sum)
    recall=tp_sum/(tp_sum+fn_sum)
    print(prec,recall)
    print(tp_sum,fp_sum,fn_sum,d_sum,t_sum)
    return "{},{},{},{},{},{},{}".format(prec,recall,tp_sum,fp_sum,fn_sum,d_sum,t_sum)



if __name__=="__main__":
    xmlPath1="/home/hyl/data/ljk/github-pro/zjai-com/data/predict_data/test_data-2018-07-30/Annotations"
    xmlPath2="/home/hyl/data/ljk/github-pro/zjai-com/data/predict_data/test_data-2018-07-30/Annotations_true"
    save_path='/home/hyl/data/ljk/github-pro/zjai-com/data/predict_data/images'
    compare_from_xml(xmlPath1,xmlPath2)

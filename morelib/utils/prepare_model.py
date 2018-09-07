# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 8/9/2018 4:34 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
import os
import tensorflow as tf

from nets.resnet_v1 import resnetv1
from nets.vgg16 import vgg16
from lib.model.config import cfg

def get_tf_model(model_dir,model_data):
    tf_model = os.path.join(model_dir, model_data)
    if not os.path.isfile(tf_model+".meta" ):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tf_model))
    return tf_model

def load_model(sess,demonet,tf_model,classes_num):
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", classes_num, tag='default', anchor_scales=cfg.ANCHOR_SCALES,anchor_ratios=cfg.ANCHOR_RATIOS)
    saver = tf.train.Saver()
    saver.restore(sess, tf_model)   #加载模型权重

    print('Loaded network {:s}'.format(tf_model))
    return saver,net

def load_forecast_files(forecast_dir):
    if os.path.exists(forecast_dir)!=1:
        os.mkdir(forecast_dir)
        os.mkdir(os.path.join(forecast_dir,"JPEGImages"))
        os.mkdir(os.path.join(forecast_dir,"Annotations_test"))
    jpg_path=os.path.join(forecast_dir,'JPEGImages')
    xml_path=os.path.join(forecast_dir,"Annotations_test")
    jpg_files=[]
    for file in os.listdir(jpg_path):
        jpg_files.append(os.path.join(jpg_path,file))
    if len(jpg_files)==0:
        print("Please load some Images in the {}!".format(jpg_path))
    return jpg_files,xml_path
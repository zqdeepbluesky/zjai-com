from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg

import tensorflow as tf
import cv2
import argparse
import os.path as osp
from nets.resnet_v1 import resnetv1
from nets.vgg16 import vgg16
from morelib.newUtils.xml_fromsg import *
from lib.datasets import pascal_voc
from morelib.test import predict


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        default='pascal_voc_0712')
    parser.add_argument('--root_dir', dest='root_dir', help='the path of the file hava stored',
                         default=osp.abspath(osp.join(osp.dirname(__file__), '..',"..","data")))
    parser.add_argument('--set_name', dest='set_name', help='the name of the classes file',
                        default="com")
    parser.add_argument('--model_dir', dest='model_dir', help='the path of  stored the model file',
                        default=osp.abspath(osp.join(osp.dirname(__file__), '..',"..","data","model")))
    parser.add_argument('--forecast_dir', dest='forecast_dir', help='prepare to forecast this picture',
                        default=osp.abspath(osp.join(osp.dirname(__file__), '..', "..", "data", "forecast")))
    args = parser.parse_args()

    return args

def get_tf_model(model_dir):
    tf_model = ''
    for files in os.listdir(model_dir):
        if files.find(".meta") != -1:
            tf_model = os.path.join(model_dir, files[:files.find(".meta")])
            break
    print(tf_model)
    if not os.path.isfile(tf_model + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tf_model + '.meta'))
    return tf_model

def load_model(demonet,tf_model,classes_num):
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", classes_num, tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tf_model)   #加载模型权重

    print('Loaded network {:s}'.format(tf_model))
    return saver,net

def load_forecast_files(forecast_dir):
    if os.path.exists(forecast_dir)!=1:
        os.mkdir(forecast_dir)
        os.mkdir(os.path.join(forecast_dir,"JPEGImages"))
        os.mkdir(os.path.join(forecast_dir,"Annotations"))
    jpg_path=os.path.join(forecast_dir,'JPEGImages')
    xml_path=os.path.join(forecast_dir,"Annotations")
    jpg_files=[]
    for file in os.listdir(jpg_path):
        jpg_files.append(os.path.join(jpg_path,file))
    if len(jpg_files)==0:
        print("Please load some Images in the {}!".format(jpg_path))
    return jpg_files,xml_path

def predict_all_images(sess,net,jpg_files,xml_path):
    for image in jpg_files:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(image))
        im = cv2.imread(image)
        result_data=predict.detect_single_image(sess, net, im,CLASSES)
        im_info={"path":image}
        im_info["width"]=im.shape[0]
        im_info["height"]=im.shape[1]
        im_info["name"]=os.path.splitext(os.path.split(image)[1])[0]
        im_info["channel"]=im.shape[2]
        save_annotations(xml_path,im_info,result_data)


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    CLASSES = pascal_voc.read_classes(os.path.join(args.root_dir,"cfgs","{}_classes.txt".format(args.set_name)))

    tf_model=get_tf_model(args.model_dir)
    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    saver,net=load_model(args.demo_net,tf_model,len(CLASSES))

    jpg_files,xml_path=load_forecast_files(args.forecast_dir)
    predict_all_images(sess,net,jpg_files,xml_path)
# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 8/9/2018 4:34 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tools import _init_paths
from model.config import cfg

import cv2
import argparse
import os.path as osp
import tensorflow as tf

from morelib.utils.xml_store import *
from lib.datasets import pascal_voc
from morelib.utils import prepare_model, predict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        default='pascal_voc_0712')
    parser.add_argument('--root_dir', dest='root_dir', help='the path of the file hava stored',
                         default=osp.join(cfg.ROOT_DIR,"data"))
    parser.add_argument('--set_name', dest='set_name', help='the name of the classes file',
                        default="com")
    parser.add_argument('--model_dir', dest='model_dir', help='the path of  stored the model file',
                        default=osp.join(cfg.ROOT_DIR, "data","model"))
    parser.add_argument('--model_data', dest='model_data', help='the name of  stored the model file',
                        default="vgg16_faster_rcnn_iter_3335000.ckpt")
    parser.add_argument('--predict_dir', dest='predict_dir', help='prepare to predict this image',
                        default=osp.join(cfg.ROOT_DIR, "data"))
    parser.add_argument('--package_data', dest='package_data', help='the test package data file name',
                        default="demo")
    parser.add_argument('--file_name', dest='file_name', help='the test jpg file name',
                        default="train_2018-07-24_1257.jpg")
    args = parser.parse_args()

    return args

def predict_test(sess,net,args):
    image_file=os.path.join(args.predict_dir,args.package_data,args.file_name)
    print(image_file)
    im = cv2.imread(image_file)
    scores, boxes = predict._detect_image(sess, net, im)
    result_data = predict.predict_image(scores, boxes, CLASSES)
    im_info = {"path": image_file}
    im_info["width"] = im.shape[0]
    im_info["height"] = im.shape[1]
    im_info["name"] = os.path.splitext(os.path.split(image_file)[1])[0]
    im_info["channel"] = im.shape[2]
    xml_path=os.path.split(image_file)[0]
    save_annotations(xml_path, im_info, result_data)


def predict_proc(sess, net, image):
    result_data = predict.predict_image(sess, net, image, CLASSES)
    return result_data

args = parse_args()
CLASSES = pascal_voc.read_classes(os.path.join(args.root_dir,"cfgs","{}_classes.txt".format(args.set_name)))

def predict_frnn(image):
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    tf_model = prepare_model.get_tf_model(args.model_dir, args.model_data)
    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    saver, net = prepare_model.load_model(sess, args.demo_net, tf_model, len(CLASSES))

    return predict_proc(sess, net, image)

def main():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    tf_model = prepare_model.get_tf_model(args.model_dir, args.model_data)
    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    saver, net = prepare_model.load_model(sess, args.demo_net, tf_model, len(CLASSES))

    predict_test(sess, net, args)

if __name__ == '__main__':
    main()
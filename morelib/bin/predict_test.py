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
from morelib.utils.xml_fromsg import *
from lib.datasets import pascal_voc
from morelib.bin import predict
from morelib.bin import predict_batch


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        default='pascal_voc_0712')
    parser.add_argument('--root_dir', dest='root_dir', help='the path of the file hava stored',
                         default=osp.join(cfg.ROOT_DIR,"data"))
    parser.add_argument('--set_name', dest='set_name', help='the name of the classes file',
                        default="com")
    parser.add_argument('--model_dir', dest='model_dir', help='the path of  stored the model file',
                        default=osp.join(cfg.ROOT_DIR, "data","model"))
    parser.add_argument('--predict_dir', dest='predict_dir', help='prepare to predict this image',
                        default=osp.join(cfg.ROOT_DIR, "data","predict_data"))
    parser.add_argument('--package_data', dest='package_data', help='the test package data file name',
                        default="test_data-2018-07-19")
    parser.add_argument('--file_name', dest='file_name', help='the test jpg file name',
                        default="test_2018-07-19_1421.jpg")
    args = parser.parse_args()

    return args


def predict_test(sess,net,args):
    image_file=os.path.join(args.predict_dir,args.package_data,args.file_name)
    print(image_file)
    im = cv2.imread(image_file)
    result_data = predict.predict_image(sess, net, im, CLASSES)
    im_info = {"path": image_file}
    im_info["width"] = im.shape[0]
    im_info["height"] = im.shape[1]
    im_info["name"] = os.path.splitext(os.path.split(image_file)[1])[0]
    im_info["channel"] = im.shape[2]
    xml_path=os.path.split(image_file)[0]
    save_annotations(xml_path, im_info, result_data)


args = parse_args()
CLASSES = pascal_voc.read_classes(os.path.join(args.root_dir,"cfgs","{}_classes.txt".format(args.set_name)))

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    tf_model=predict_batch.get_tf_model(args.model_dir)
    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    saver,net=predict_batch.load_model(sess,args.demo_net,tf_model,len(CLASSES))

    predict_test(sess,net,args)
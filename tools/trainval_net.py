# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 8/9/2018 4:34 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pprint
import numpy as np

from tools import _init_paths
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir,load_setting_cfg
from model.train_val import train_net
from nets.mobilenet_v1 import mobilenetv1
from nets.resnet_v1 import resnetv1
from nets.vgg16 import vgg16
from tools.preprocessing import calc_roidb
from lib.extra_utils.serialize_argparse import SerializeArgparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def parse_args():
    """
    Parse input arguments
    """
    parser = SerializeArgparse(description='FRCNN trainer.')
    parser.add_argument('--gpu', dest='gpu',
                        help='use gpu id',
                        default='0', type=str)
    parser.add_argument('--snapshot', dest='snapshot',
                        help='whether fine tune',
                        default=False, type=bool)
    parser.add_argument('--snapshot_dir', dest='snapshot_dir',
                        help='fine tune dir',
                        default='', type=str)
    parser.add_argument('--load_args_json', dest='load_args_json',
                        help='if load args json',
                        default=False, type=bool)
    parser.add_argument('--args_json_dir', dest='args_json_dir',
                        help='args json path',
                        default='data/args_parse/args_json', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='experiments/cfgs/vgg16.yml', type=str)
    parser.add_argument('--weight', dest='weight',
                        help='initialize with pretrained model weights',
                        default='data/imagenet_weights/vgg16.ckpt',
                        type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--imdbval', dest='imdbval_name',
                        help='dataset to validate on',
                        default='voc_2007_test', type=str)
    parser.add_argument('--epochs', dest='epochs',
                        help='epoch of iteration to train',
                        default=50, type=int)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=80000, type=int)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default=None, type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152, mobile',
                        default='vgg16', type=str)
    parser.add_argument('--use_extra_test_data', dest='use_extra_test_data',
                        help='whether want to use test data to test the model',
                        default=True, type=bool)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=['ANCHOR_SCALES', '[8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'TRAIN.STEPSIZE', '[2400000]'],
                        type=list)
    parser.add_argument('--package_name', dest='package_name',
                        help='train_data1,train_data2,train_data3',
                        default=['all_train_data_resize2'], type=list)
    parser.add_argument('--com_classes', dest='com_classes',
                        help='use com_classes file name',
                        default='com_classes_21.txt', type=str)
    parser.add_argument('--extra_test_dir', dest='extra_test_dir',
                        help='train_data1,train_data2,train_data3',
                        default='data/predict_data', type=str)
    parser.add_argument('--extra_test_package', dest='extra_test_package',
                        help='train_data1,train_data2,train_data3',
                        default=['all_train_data_resize2'], type=list)
    parser.add_argument('--use_early_stop', dest='use_early_stop',
                        help='whether use early stop',
                        default=False, type=bool)
    parser.add_argument('--stop_iter', dest='stop_iter',
                        help='stop at which iter',
                        default=4000000, type=str)
    parser.add_argument('--stop_loss', dest='stop_loss',
                        help='stop at which loss level',
                        default=0.001, type=str)
    return parser

def load_base_network():
    if args.net == 'vgg16':
        net = vgg16()
    elif args.net == 'res50':
        net = resnetv1(num_layers=50)
    elif args.net == 'res101':
        net = resnetv1(num_layers=101)
    elif args.net == 'res152':
        net = resnetv1(num_layers=152)
    elif args.net == 'mobile':
        net = mobilenetv1()
    else:
        raise NotImplementedError
    return net

def prepare_datas(package_name):
    # train set
    # imdb, roidb = combined_roidb(args.imdb_name)
    imdb, roidb = calc_roidb(args.imdb_name,package_name)
    logger.info('{:d} roidb entries'.format(len(roidb)))

    # also add the validation set, but with no flipping images
    hor_orgflip = cfg.TRAIN.USE_HOR_FLIPPED
    ver_orgflip = cfg.TRAIN.USE_VER_FLIPPED
    org_bright_adjuest = cfg.TRAIN.BRIGHT_ADJUEST
    org_rotate_adjuest = cfg.TRAIN.ROTATE_ADJUEST
    org_shift_adjuest = cfg.TRAIN.SHIFT_ADJUEST
    org_zoom_adjuest = cfg.TRAIN.ZOOM_ADJUEST
    org_crop_images = cfg.TRAIN.CROP_IMAGES
    cfg.TRAIN.USE_HOR_FLIPPED = False
    cfg.TRAIN.USE_VER_FLIPPED = False
    cfg.TRAIN.BRIGHT_ADJUEST = False
    cfg.TRAIN.ROTATE_ADJUEST = False
    cfg.TRAIN.SHIFT_ADJUEST = False
    cfg.TRAIN.ZOOM_ADJUEST = False
    cfg.TRAIN.CROP_IMAGES = False

    _, valroidb = calc_roidb(args.imdbval_name,package_name)
    logger.info('{:d} validation roidb entries'.format(len(valroidb)))
    cfg.TRAIN.USE_HOR_FLIPPED = hor_orgflip
    cfg.TRAIN.USE_VER_FLIPPED = ver_orgflip
    cfg.TRAIN.BRIGHT_ADJUEST = org_bright_adjuest
    cfg.TRAIN.ROTATE_ADJUEST = org_rotate_adjuest
    cfg.TRAIN.SHIFT_ADJUEST = org_shift_adjuest
    cfg.TRAIN.ZOOM_ADJUEST = org_zoom_adjuest
    cfg.TRAIN.CROP_IMAGES = org_crop_images

    return imdb, roidb, valroidb
def print_args(args,verbose=1):
    print('*'*20)
    print('Called with key args:')
    if verbose==1:
        show_key_args(args)
    if verbose==2:
        print(args)
    if verbose==3:
        print(args)
        pprint(cfg)
    print('*' * 20)

def show_key_args(args):
    train_package = [os.path.join(cfg.ROOT_DIR,'data','train_data', package) for package in args.package_name]
    print('--input training package : {}'.format(train_package))
    print('--whether use fine tune : {}'.format(args.snapshot))
    if args.snapshot:
        print('--use fine tune model : {}'.format(args.snapshot_dir))
    print('--use net : {}'.format(args.net))
    print('--use classes file : {}'.format(args.com_classes))
    test_package = [os.path.join(cfg.ROOT_DIR, 'data', 'train_data', package) for package in args.package_name]
    print('--input test package : {}'.format(test_package))
    if args.use_extra_test_data:
        extra_test_package = [os.path.join(cfg.ROOT_DIR, args.extra_test_dir, package) for package in
                              args.extra_test_package]
        print('--input extra test package : {}'.format(extra_test_package))
    else:
        print('--use extra test package : {}'.format(args.use_extra_test_data))
def prepare_params():
    # output directory where the models are saved
    output_dir = get_output_dir(imdb, args.tag,args.package_name)
    logger.info('Output will be saved to `{:s}`'.format(output_dir))

    # tensorboard directory where the summaries are saved during training
    tb_dir = get_output_tb_dir(imdb, args.tag)
    logger.info('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    return output_dir, tb_dir


if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()

    args=load_setting_cfg(args)
    print_args(args,verbose=1)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.CLASSES_FILE = args.com_classes

    args=parser.reload_or_save_args(cfg,args)

    # print('Using config:')
    # pprint.pprint(cfg)

    np.random.seed(cfg.RNG_SEED)

    # load network
    base_net = load_base_network()

    # train and valid data
    imdb, roidb, valroidb = prepare_datas(args.package_name)

    # output data directory
    output_dir, tb_dir = prepare_params()

    train_net(args,base_net, imdb, roidb, valroidb, output_dir, tb_dir, pretrained_model=args.weight, max_iters=args.max_iters*args.epochs)
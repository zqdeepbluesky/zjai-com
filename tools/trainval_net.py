# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pprint
import numpy as np

from tools import _init_paths
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from model.train_val import train_net
from nets.mobilenet_v1 import mobilenetv1
from nets.resnet_v1 import resnetv1
from nets.vgg16 import vgg16
from tools.preprocessing import calc_roidb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def print_args(args):
    if args is None:
        print('args error')

    print('cfg_file:', args.cfg_file)
    print('set_cfgs:', args.set_cfgs)
    print('tag:', args.tag)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='experiments/cfgs/vgg16.yml', type=str)
    parser.add_argument('--weight', dest='weight',
                        help='initialize with pretrained model weights',
                        default=os.path.join(cfg.ROOT_DIR, 'data/imagenet_weights/vgg16.ckpt'),
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
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=['ANCHOR_SCALES', '[8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'TRAIN.STEPSIZE', '[2400000]'],
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    print('*'*20)
    print_args(args)

    return args

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

def prepare_datas():
    # train set
    # imdb, roidb = combined_roidb(args.imdb_name)
    imdb, roidb = calc_roidb(args.imdb_name)
    logger.info('{:d} roidb entries'.format(len(roidb)))

    # also add the validation set, but with no flipping images
    orgflip = cfg.TRAIN.USE_FLIPPED
    cfg.TRAIN.USE_FLIPPED = False
    _, valroidb = calc_roidb(args.imdbval_name)
    logger.info('{:d} validation roidb entries'.format(len(valroidb)))
    cfg.TRAIN.USE_FLIPPED = orgflip

    return imdb, roidb, valroidb

def prepare_params():
    # output directory where the models are saved
    output_dir = get_output_dir(imdb, args.tag)
    logger.info('Output will be saved to `{:s}`'.format(output_dir))

    # tensorboard directory where the summaries are saved during training
    tb_dir = get_output_tb_dir(imdb, args.tag)
    logger.info('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    return output_dir, tb_dir

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    np.random.seed(cfg.RNG_SEED)

    # load network
    base_net = load_base_network()

    # train and valid data
    imdb, roidb, valroidb = prepare_datas()

    # output data directory
    output_dir, tb_dir = prepare_params()

    train_net(base_net, imdb, roidb, valroidb, output_dir, tb_dir, pretrained_model=args.weight, max_iters=args.max_iters*args.epochs)
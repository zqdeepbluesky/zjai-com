# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 8/9/2018 4:34 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tools.serialize_argparse import SerializeArgparse
import os
from tools import _init_paths
from model.config import cfg

parser = SerializeArgparse(description='FRCNN trainer.')
parser.add_argument('-name',dest='name', type=str, default=os.path.join(cfg.ROOT_DIR, "data/args_parse") , help="Directory to save args")
parser.add_argument('-learningRate',dest='learningRate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('-gpu', dest='gpu',type=str, default="0", help='Train on this GPU(s)')

opt=parser.parse_args()

if not os.path.isdir(opt.name):
	os.makedirs(opt.name)

opt = parser.load(opt.name+"/args.json")
parser.save(opt.name+"/args.json")
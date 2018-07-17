# -*- coding: utf-8 -*-
# @Time    : 7/17/2018 10:57 AM
# @Author  : sunyonghai
# @File    : preprocessing.py
# @Software: ZJ_AI
import logging

import datasets.factory
import model.train_val
from model.config import cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def combined_roidb(imdb_name):
    logger.warning("{}".format('The `combined_roidb` function is deprecat and will be removed after 08/2018. Use instead `calc_roidb`'))
    return calc_roidb(imdb_name)

def calc_roidb(imdb_name):

    imdb = datasets.factory.get_imdb(imdb_name)
    logger.info('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)

    logger.info('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = model.train_val.get_training_roidb(imdb)

    return imdb, roidb

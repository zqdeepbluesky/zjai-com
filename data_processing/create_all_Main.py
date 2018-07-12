#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 4/3/2018 10:37 AM 
# @Author : sunyonghai 
# @File : create_all_Main.py 
# @Software: ZJ_AI
# =========================================================
import os

from config import ROOT_HOME
from create_Main import create_sub_Main


def _read_txt(path):
    with open(path, 'r') as f:
        data = f.readlines()

    return data

def _write_txt(path,data):
    # path = os.path.join(ROOT_HOME, 'ImageSets/Main/trainval.txt')
    with open(path, 'w+') as f:
        f.writelines(data)

    print("Save the data to {}, number:{}".format(path, len(data)))
    return True

def _read_trainval(data_dir):
    txt_path = os.path.join(data_dir, 'ImageSets/Main/trainval.txt')
    data = _read_txt(txt_path)
    return data

def _read_test(data_dir):
    path_txt = os.path.join(data_dir, 'ImageSets/Main/test.txt')
    data = _read_txt(path_txt)
    return data

def create_all_Main(dirs):
    data_dirs = [os.path.join(ROOT_HOME, s) for s in dirs]
    trainvals = []
    tests = []
    for data_dir in data_dirs:
        trainvals.extend(_read_trainval(data_dir))
        tests.extend(_read_test(data_dir))

    trainval_path = os.path.join(ROOT_HOME, 'data/all_data/ImageSets/Main/trainval.txt')
    test_path = os.path.join(ROOT_HOME, 'data/all_data/ImageSets/Main/test.txt')

    _write_txt(trainval_path, trainvals)
    _write_txt(test_path, tests)


if __name__ == '__main__':
    # dirs = ['data/train_data-2018-3-7', 'data/train_data-2018-3-16']
    dirs = ['data/train_data-2018-03-07', 'data/train_data-2018-03-16', 'data/train_data-2018-03-19', 'data/train_data-2018-03-29','data/train_data-2018-03-30','data/train_data-2018-04-02']
    # dirs = ['data/train_data-2018-04-02']
    create_sub_Main(dirs)
    create_all_Main(dirs)



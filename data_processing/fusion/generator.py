# -*- coding: utf-8 -*-
# @Time    : 6/25/2018 4:23 PM
# @Author  : sunyonghai
# @File    : generator.py
# @Software: ZJ_AI
import itertools
import json
import random
import threading

from data_processing.fusion import fusion_utils
import numpy as np
import os

data_path = '/home/syh/tf-faster-rcnn/data/fusion/mask'
bg_path = '/home/syh/tf-faster-rcnn/data/fusion/bg'

def read_classes(path):
        with open(path, 'r') as f:
            # names_to_labels
            data = json.load(f)
            # pprint('mapping info:', labels_to_names)
        return data


class Generator(object):
    def  __init__(self, data_path, bg_path, classes, batch_size = 6, target_size=(224, 224)):
        self.image_instances = []
        self.target_size = target_size
        self.batch_size = batch_size
        self.group_index = 0
        self.lock = threading.Lock()
        self.groups = []
        self.classes = classes
        self.labels = {}

        self.init(data_path, bg_path)
        self.group_images()

    def init(self, data_path,bg_path):
        for key, value in self.classes.items():
            self.labels[value] = key

        self.image_instances = self.load_objects(data_path)
        self.bg_gen = self.load_background(bg_path)
        # for subdir in os.listdir(data_path):
        #     sub_path = os.path.join(data_path, subdir)
        #     for file in os.listdir(sub_path):
        #         self.image_instances.append(os.path.join(sub_path, file))

    def load_objects(self, path):
        file_paths = []
        for root, _, files in os.walk(path, topdown=False):
            file_paths.extend([os.path.join(root, name) for name in files])

        random.shuffle(file_paths)

        return file_paths

    def load_background(self, path):
        bgs = os.listdir(path)
        bg_paths = [os.path.join(path, name) for name in bgs]

        random.shuffle(bg_paths)
        return itertools.cycle(bg_paths)

    def size(self):
        return len(self.image_instances)

    def num_classes(self):
        return len(self.classes)

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def group_images(self):
        order = list(range(self.size()))
        random.shuffle(order)

        for i in range(0, len(order), self.batch_size):
            groups = []
            for x in range(i, i + self.batch_size):
                groups.append(order[x % len(order)])  # 防止最后一个batch_size 不够数， x也不能越界

            self.groups.append(groups)

    def resize_image(self, image):
        return image

    def preprocess_image(self,image):
        return image

    def random_transform_group_entry(self,image):
        return image

    def preprocess_group_entry(self, image):
        # # resize image
        # image = self.resize_image(image)
        #
        # # preprocess the image
        # image = self.preprocess_image(image)
        #
        # # randomly transform image
        # image = self.random_transform_group_entry(image)

        return image

    def preprocess_group(self, image_group):
        for index, image in enumerate(image_group):

            # preprocess a single group entry
            image = self.preprocess_group_entry(image)

            # copy processed data back to group
            image_group[index] = image

        return image_group

    def load_image(self, image_index):
        path = self.image_instances[image_index]
        return fusion_utils.read_image(path, 'RGBA')     # RGB => BGR tensorflow train format.

    def load_image_group(self,group):
        return [self.load_image(image_index) for image_index in group]

    def load_label(self,image_index):
        path = self.image_instances[image_index]
        name = os.path.basename( os.path.dirname(path))
        label = self.name_to_label(name)
        y = np.zeros(self.num_classes())
        y[label] = 1

        return y

    def load_label_group(self, group):
        return [self.load_label(image_index) for image_index in group]

    def load_bg(self):
        return fusion_utils.read_image(next(self.bg_gen), 'RGB')

    def compute_inputs(self, image_group):
        # # get the max image shape
        # max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        #
        # # construct an image batch object
        # image_batch = np.zeros((self.batch_size,) + max_shape, dtype=np.float32)
        #
        # # copy all images to the upper left part of the image batch object
        # for image_index, image in enumerate(image_group):
        #     image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image
        #
        # for image_index, image in enumerate(image_group):
        #     image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_group

    def compute_targets(self,label_group):
        label_batch = np.zeros((self.batch_size,) + (label_group[0].shape[0],), dtype=np.float32)
        for index, label in enumerate(label_group):
            label_batch[index, ...] = label

        return label_batch

    def compute_input_output(self, group):
        # load images and annotations
        image_group  = self.load_image_group(group)
        label_group =  self.load_label_group(group)

        image_group = self.preprocess_group(image_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(label_group)

        return inputs, targets

    def __len__(self):
        return len(self.image_instances)

    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0:
                # shuffle groups at start of epoch
                random.shuffle(self.groups)
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)+ (self.load_bg(),)


if __name__ == '__main__':
    classes = read_classes('/home/syh/tf-faster-rcnn/data_processing/fusion/mapping_all.json')
    gen = Generator(data_path, bg_path, classes)
    x, y, bg = gen.next()
    print(x.shape, y.shape, bg.shape)
    print(y)

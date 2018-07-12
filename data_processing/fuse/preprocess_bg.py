# -*- coding: utf-8 -*-
# @Time    : 5/31/2018 10:06 AM
# @Author  : sunyonghai
# @File    : preprocess_bg.py
#multiprocessing "handle_background"
# @Software: ZJ_AI
import multiprocessing
import multiprocessing.dummy
import os
import cv2
import numpy as np
import PIL.Image

def resize_image(img, min_side=800, max_side=1333):
    (rows, cols, _) = img.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale

# read image from disk
def read_image_rgb(path):
    try:
        image = np.asarray(PIL.Image.open(path).convert('RGB'))
    except Exception as ex:
        print('{}'.format(path))

    return image.copy()

# laod image
def load_image(path):
    return read_image_rgb(path)

# save image
def save_image(img, path):
    try:
        cv2.imwrite(path, img)
        print(path)
    except Exception as ex:
        print(ex)

# get the name of bg
def get_image_path():
    lock.acquire()
    global idx
    path = os.path.join(output, '{}_2018-05-31_{}.jpg'.format(prefix, idx))
    idx +=1
    lock.release()
    return path

# get all data paths
def get_data(input):
    try:
        files = os.listdir(input)
        data_paths = [os.path.join(input, file) for file in files]
    except Exception as ex:
        print(ex)

    return data_paths

# start to process
def process(image_path):
    image = load_image(image_path)
    img, scale =  resize_image(image)

    output_path = get_image_path()
    save_image(img, output_path)

# mutil process
def mutil_process():
    all_data = get_data(input)
    cpus = os.cpu_count() // 2
    p = multiprocessing.pool.Pool(cpus)
    p.map_async(process, all_data)

    p.close()
    p.join()

# single process
def single_process():
    data_paths = get_data(input)
    for item in data_paths:
        process(item)

def main():
    single_process()

# global variable
lock = multiprocessing.Lock()
prefix = 'bg'
idx = 10000
input = 'F:\\bg\\Background\\origin2'
output = 'F:\\bg\\Background\\after2'

if __name__ == '__main__':
    main()
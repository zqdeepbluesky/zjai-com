import matplotlib.pyplot as plt
import os
from tools import _init_paths
from model.config import cfg
from datasets import pascal_voc

import pickle
import PIL
from PIL import Image
import argparse
import random
import cv2
import numpy as np
from skimage import exposure

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--cache_dir', dest='cache_dir', help='Network to use [vgg16 res101]',
                        default=os.path.join(cfg.ROOT_DIR, "data", "cache"))
    parser.add_argument('--roidb_name', dest='roidb_name', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        default='fusion_2018-08-06_voc_2007_trainval_enhance_roidb_00001000.pkl')
    parser.add_argument('--root_dir', dest='root_dir', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        default=os.path.join(cfg.ROOT_DIR,"data"))
    args = parser.parse_args()

    return args

def read_image_rgb(path):
    try:
        image = np.asarray(PIL.Image.open(path).convert('RGB'))
    except Exception as ex:
        print('{}'.format(path))

    return image.copy()

def get_roidb(cache_dir,roidb_name):
    roidb_file=os.path.join(cache_dir,roidb_name)
    with open(roidb_file, 'rb') as fid:
        try:
            roidb = pickle.load(fid)
        except:
            roidb = pickle.load(fid, encoding='bytes')
    return roidb

def _bright_adjuest(im,gamma):
    im = exposure.adjust_gamma(im, gamma)
    return im

from math import *
def _rotate_image(img,angle):
    height,width=img.shape[0],img.shape[1]
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步，目前不懂为什么加这步
    matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))
    return imgRotation

def _translateit(image, offset, isseg=False):
    from scipy.ndimage.interpolation import shift
    order = 0
    return shift(image, (int(offset[0]), int(offset[1]), 0), order=order, mode='nearest')

def draw_roidb(roidb):
    from lib.datasets import data_augment
    print(roidb)
    image_path=roidb['image']
    print(image_path)
    boxes=roidb['boxes']
    classes=roidb['gt_classes']
    im=read_image_rgb(image_path)
    if 'ver_flipped' in roidb and roidb['ver_flipped']==True:
        im = im[::-1, :, :]
    if 'hor_flipped' in roidb and roidb['hor_flipped']==True:
        im = im[:, ::-1, :]
    if 'bright_scala' in roidb and roidb['bright_scala'] != 1:
        im = _bright_adjuest(im, roidb['bright_scala'])
    if 'rotate_angle' in roidb and roidb['rotate_angle'] != 0:
        im = _rotate_image(im, roidb['rotate_angle'])
    if 'shift_x' in roidb and 'shift_y' in roidb:
        offset=(int(roidb['shift_x']),int(roidb['shift_y']))
        im = _translateit(im, offset)
    if 'zoom_x' in roidb and 'zoom_y' in roidb:
        factor_x, factor_y = int(roidb['zoom_x']), int(roidb['zoom_y'])
        im = data_augment._zoom_image(im, factor_x, factor_y)
    if 'position' in roidb and 'crop_size_width' in roidb and 'crop_size_height' in roidb:
        crop_size = (roidb['crop_size_width'], roidb['crop_size_height'])
        scale = cfg.TRAIN.RESIZE_SCALE
        position = roidb['position']
        im = data_augment.random_crop_image(im, crop_size, scale, position)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(len(classes)):
        bbox=boxes[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s}'.format(CLASSES[int(classes[i])]),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.show()



args=parse_args()
CLASSES = pascal_voc.read_classes(os.path.join(cfg.ROOT_DIR,"experiments",'classes_cfgs',"com_classes_169.txt"))

if __name__=="__main__":
    roidb=get_roidb(args.cache_dir,args.roidb_name)
    for i in range(10):
        num=random.randint(0,len(roidb))
        draw_roidb(roidb[num])
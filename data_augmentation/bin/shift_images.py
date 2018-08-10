from scipy.ndimage.interpolation import shift
import cv2
import os
import argparse
from math import *
from data_augmentation.utils.xml_utils import show_object_cv_box
from data_augmentation.utils import io_utils, xml_store

from tools import _init_paths
from model.config import cfg


def translateit(image, offset, isseg=False):
    order = 0
    return shift(image, (int(offset[0]), int(offset[1]), 0), order=order, mode='nearest')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--data_dir', dest='data_dir', help='prepare to compare this image and xml',
                        default=os.path.join(cfg.ROOT_DIR, 'data', 'train_data'))
    parser.add_argument('--package_dir', dest='package_dir', help='the compare data file name',
                        default="train_data-2018-03-07")
    parser.add_argument('--offset', dest='offset', help='how many the num you want to shift',
                        default=(-200, -200))
    # (100,100)代表向右、想下平移100，（-100，-100）代表向左向上平移100
    args = parser.parse_args()

    return args

def adjuest_obj_info(obj_info, offset, size):
    new_obj_info=[]
    for obj in obj_info:
        classes, _, xmin, ymin, xmax, ymax = obj.split(",")
        xmin = fix_new_key(int(int(xmin) + offset[0]),offset[0],size[1])
        ymin = fix_new_key(int(int(ymin) + offset[1]),offset[1],size[0])
        xmax = fix_new_key(int(int(xmax) + offset[0]),offset[0],size[1])
        ymax = fix_new_key(int(int(ymax) + offset[1]),offset[1],size[0])
        if xmax-xmin!=0 and ymax-ymin!=0:
            new_obj_info.append("{},1,{},{},{},{}".format(classes,xmin,ymin,xmax,ymax))
    return new_obj_info



def fix_new_key(key,offset,bound):
    if offset>=0:
        key=min(key,bound)
    else:
        key=max(0,key)
    return key


def shift_and_save_images(offset, data_dir):
    jpg_path = os.path.join(data_dir, "JPEGImages")
    annot_path = os.path.join(data_dir, "Annotations")
    # io_utils.mkdir(os.path.join(data_dir + '_shift_{}_{}'.format(offset[0],offset[1]), 'JPEGImages'))
    for i in os.listdir(jpg_path):
        print(i)
        a, b = os.path.splitext(i)
        if b == ".jpg":
            img_path = os.path.join(jpg_path, i)
            img = cv2.imread(img_path)

            xml_path = os.path.join(annot_path, a + ".xml")
            object_infos = xml_store.get_object_infos_from_xml(xml_path)
            show_object_cv_box(object_infos, img)
            img_shift = translateit(img, offset)
            new_obj_infos = adjuest_obj_info(object_infos, offset, img.shape[:2])
            show_object_cv_box(new_obj_infos, img_shift)


args = parse_args()

if __name__ == "__main__":
    shift_and_save_images(args.offset, os.path.join(args.data_dir, args.package_dir))

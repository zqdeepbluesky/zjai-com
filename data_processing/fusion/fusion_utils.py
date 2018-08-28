# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 8/22/2018 5:27 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import PIL.Image
import os
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

# read image from disk
def read_image(path, mode=None):
    try:
        image = cv2.imread(path,cv2.IMREAD_UNCHANGED) #np.asarray(PIL.Image.open(path).convert(mode))
    except Exception as ex:
        print('{}'.format(path))

    return image.copy()

# laod image
def load_image(path):
    return read_image(path)

def resize_image(image, size):
    # resize the image with the  size
    img = cv2.resize(image, size)
    return img

def save_image(out_path, image):
    # try:
    #     image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    # except Exception as ex:
    #     print(out_path)
    #
    try:
        if out_path is not None:
            dir = os.path.dirname(out_path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            cv2.imwrite(out_path, image)
    except Exception as ex:
        print(ex)

def create_png(mask_img, origin_img):
    '''
    :param mask_img:
    :param origin_img:
    :param out_path:
    :return:
    '''
    result = np.zeros((origin_img.shape[0], origin_img.shape[1], 4), np.uint8)
    result[:, :, 3] = mask_img[:, :, 0].copy()
    result[:, :, 0:3][np.where(mask_img != 0)] = origin_img[np.where(mask_img != 0)]

    # save_image(out_path, result)
    return result

def find_bbox(mask):
    sum_each_col = mask.astype(np.double).sum(0).astype(np.uint8)
    sum_each_row = mask.astype(np.double).sum(1).astype(np.uint8)
    col_nonzero_index = np.nonzero(sum_each_col)[0].tolist()
    row_nonzero_index = np.nonzero(sum_each_row)[0].tolist()

    if(col_nonzero_index!=[] and row_nonzero_index!=[]):
        minx = col_nonzero_index[0]
        maxx = col_nonzero_index[-1]+1#minx + width
        miny = row_nonzero_index[0]
        maxy = row_nonzero_index[-1]+1#minx + height
        return [(minx,miny),(maxx,maxy)]
    else:
        return [(),()]

def composite_bg(bg, obj_png):
    # obj_height, obj_width = obj_png.shape[0:2]
    img = bg.copy()

    alpha = obj_png[:, :, 3]
    img[:, :, :][alpha!=0] = obj_png[:,:,:3][alpha!=0]

    # save_image(out_path, img)
    return img

def paste_obj(bg_buffer, obj_png, box):
    alpha = obj_png[:,:,3]
    bg_buffer[box[0]:box[2], box[1]:box[3],:][alpha!=0] = obj_png[:,:,:][alpha!=0]
    return bg_buffer

#
# def paste_obj(obj_png, bg_size):
#     obj_height, obj_width = obj_png.shape[0:2]
#     bg_buffer = np.zeros((bg_size+ (4,)), np.uint8)
#
#     obj_alpha = obj_png[:,:,3]
#     bg_buffer[0:obj_height,0:obj_width ,:][obj_alpha!=0] = obj_png[:,:,:][obj_alpha!=0]
#     return bg_buffer


# def pasteObj(bg, obj_png):
#     obj_height, obj_width = obj_png.shape[0:2]
#     result = resize_image(bg, obj_height, obj_width)
#
#     obj_alpha = obj_png[:,:,3]
#     # for c in range(3):
#     result[0:obj_height,0:obj_width ,:][obj_alpha!=0] = obj_png[:,:,:][obj_alpha!=0]
#     return result
#
# root_path = '/home/syh/tf-faster-rcnn/data/fusion/'
#
# if __name__ == '__main__':
#     path = os.path.join(root_path, "mask/aebs-aebsntbbt-dz-hhkw-120g/0.png")
#     bg_path = os.path.join(root_path,'bg/bg_2018-05-30_10466.jpg')
#
#     obj_mask_path = os.path.join(root_path,'obj_165_mask/aebs-aebsntbbt-dz-hhkw-120g/0.png')
#
#     obj_im =  read_image_rgb(path, 'RGBA')
#     bg_im =  read_image_rgb(bg_path, 'RGB')
#     obj_mask_im =  read_image_rgb(obj_mask_path, 'RGBA')
#     img = cv2.imread(obj_mask_path, cv2.IMREAD_UNCHANGED)
#     bg_im = resize_image(bg_im, obj_im.shape[:2][::-1])
#
#     output_path = os.path.join(root_path,'output/JPEGImages/bg_2018-05-30_10466.jpg')
#     bg_obj = pasteObj(bg_im, obj_im)
#     cv2.imwrite(output_path, bg_obj)
#     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # print(im.shape)

def make_xml(im_info, boxes,labels):
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = im_info.name + '.' +im_info.image_extension

    node_path = SubElement(node_root, 'path')
    node_path.text = im_info.path

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text =str(im_info.width)

    node_height = SubElement(node_size, 'height')
    node_height.text =str(im_info.height)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(im_info.channel)

    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'

    for b,label in zip(boxes, labels):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        caption = "{}".format(label)
        node_name.text = caption

        node_pose = SubElement(node_object, 'pose')
        node_pose.text = 'Unspecified'

        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'

        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(b[0]))

        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(b[1]))

        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(b[2]))

        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(b[3]))

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)

    return dom

def save_annotations(save_dir, im_info, boxes, labels):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dom = make_xml(im_info, boxes,labels )
    xml_path = os.path.join(save_dir, im_info.name + '.xml')
    with open(xml_path, 'w+') as f:
        dom.writexml(f, addindent='', newl='', encoding='utf-8')
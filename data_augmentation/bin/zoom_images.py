import PIL
import cv2
import numpy as np
import os
import argparse
from scipy.ndimage import interpolation

from tools import _init_paths
from model.config import cfg
from morelib.utils import xml_store
from data_augmentation.utils.xml_utils import show_object_cv_box


def zoom_images_and_save(scales,data_dir):
    jpg_path=os.path.join(data_dir,"JPEGImages")
    annot_path=os.path.join(data_dir,"Annotations")

    for scale in scales:
        # io_utils.mkdir(os.path.join( data_dir+'_zoom_{}'.format(scale),'JPEGImages'))
        for i in os.listdir(jpg_path):
            a, b = os.path.splitext(i)
            if b == ".jpg":
                img_path = os.path.join(jpg_path, i)
                img = cv2.imread(img_path)

                xml_path = os.path.join(annot_path, a + ".xml")
                object_infos = xml_store.get_object_infos_from_xml(xml_path)
                show_object_cv_box(object_infos, img)
                zoom_img = zoom_image(img,scale[0],scale[1])
                new_object_infos=zoom_xml(object_infos,img,scale)
                show_object_cv_box(new_object_infos, zoom_img)
                break

                new_img_name=a + "_zoom_{}_{}".format(scale[0],scale[1]) + ".jpg"
                new_img_path=os.path.join( data_dir+'_zoom_{}_{}'.format(scale[0],scale[1]),'JPEGImages', new_img_name)


                # print(object_infos)
                # show_object_cv_box(object_infos, rotated_img)
                # print(rotated_img.shape)
                im_info=xml_utils.create_image_info(new_img_name,new_img_path,zoom_img.shape[1],zoom_img.shape[0],zoom_img.shape[2])
                new_xml_path=os.path.join( data_dir+'_zoom_{}_{}'.format(scale[0],scale[1]),'Annotations')


                cv2.imwrite(new_img_path, rotated_img)
                xml_store.save_annotations(new_xml_path, im_info, object_infos_rotate)

def zoom_image(image, factor_x,factor_y, isseg=False):
    order = 0 if isseg == True else 3
    newimg = interpolation.zoom(image, (float(factor_y), float(factor_x), 1.0), order=order, mode='nearest')
    return newimg

def zoom_xml(object_infos,img,scale):
    new_obj_infos=[]
    for object_info in object_infos:
        class_name,_,xmin, ymin, xmax, ymax = object_info.split(",")
        min_x = int(int(xmin)*scale[0])
        min_y = int(int(ymin) * scale[1])
        max_x = int(int(xmax) * scale[0])
        max_y = int(int(ymax) * scale[1])

        new_obj_infos.append("{},1,{},{},{},{}".format(class_name,min_x,min_y,max_x,max_y))
    return new_obj_infos


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--data_dir', dest='data_dir', help='prepare to compare this image and xml',
                        default=os.path.join(cfg.ROOT_DIR,'data','train_data'))
    parser.add_argument('--package_dir', dest='package_dir', help='the compare data file name',
                        default="train_data-2018-03-07")
    parser.add_argument('--zoom_scale', dest='zoom_scale', help='how many the num you want to split',
                        default=[(1,2)])
    args = parser.parse_args()

    return args

args = parse_args()


if __name__=="__main__":
    zoom_images_and_save(args.zoom_scale,os.path.join(args.data_dir,args.package_dir))

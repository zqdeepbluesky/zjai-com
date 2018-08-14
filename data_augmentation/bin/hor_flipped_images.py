import os
import argparse
import cv2
from data_augmentation.utils.xml_utils import show_object_cv_box
from data_augmentation.utils import xml_store, io_utils
from tools import _init_paths
from model.config import cfg
from data_augmentation.utils import xml_utils


def ver_flipped_images_and_save(use_hor_flipped, data_dir):
    jpg_path = os.path.join(data_dir, "JPEGImages")
    annot_path = os.path.join(data_dir, "Annotations")

    if use_hor_flipped:
        # io_utils.mkdir(os.path.join( data_dir+'_ver_flipped','JPEGImages'))
        for i in os.listdir(jpg_path):
            a, b = os.path.splitext(i)
            if b == ".jpg":
                img_path = os.path.join(jpg_path, i)
                img = cv2.imread(img_path)

                xml_path = os.path.join(annot_path, a + ".xml")
                object_infos = xml_store.get_object_infos_from_xml(xml_path)
                show_object_cv_box(object_infos, img)
                hor_flipped_img = hor_flipped_image(img)
                new_object_infos = hor_flipped_xml(object_infos, img)
                show_object_cv_box(new_object_infos, hor_flipped_img)

                break
                new_img_name = a + "_hor_flipped.jpg"
                new_img_path = os.path.join(data_dir + '_hor_flipped', 'JPEGImages', new_img_name)

                im_info = xml_utils.create_image_info(new_img_name, new_img_path, hor_flipped_img.shape[1],
                                                      hor_flipped_img.shape[0], hor_flipped_img.shape[2])
                new_xml_path = os.path.join(data_dir + '_hor_flipped', 'Annotations')

                cv2.imwrite(new_img_path, hor_flipped_img)
                xml_store.save_annotations(new_xml_path, im_info, new_object_infos)

def hor_flipped_image(im):
    im = im[:, ::-1, :]
    return im


def hor_flipped_xml(object_infos, img):
    new_obj_infos = []
    width = img.shape[1]
    height = img.shape[0]
    for object_info in object_infos:
        class_name, _, xmin, ymin, xmax, ymax = object_info.split(",")
        min_x = width - int(xmax)
        max_x = width - int(xmin)
        min_x = 0 if min_x > max_x else min_x
        new_obj_infos.append("{},1,{},{},{},{}".format(class_name, min_x, ymin, max_x, ymax))
    return new_obj_infos


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--data_dir', dest='data_dir', help='prepare to compare this image and xml',
                        default=os.path.join(cfg.ROOT_DIR, 'data', 'train_data'))
    parser.add_argument('--package_dir', dest='package_dir', help='the compare data file name',
                        default="train_data-2018-03-07")
    parser.add_argument('--use_hor_flipped', dest='use_hor_flipped', help='how many the num you want to split',
                        default=True)
    args = parser.parse_args()

    return args


args = parse_args()

if __name__ == "__main__":
    ver_flipped_images_and_save(args.use_hor_flipped, os.path.join(args.data_dir, args.package_dir))

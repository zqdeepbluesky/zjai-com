import os
import argparse
import cv2
from data_augmentation.utils.xml_utils import show_object_cv_box
from data_augmentation.utils import xml_store, io_utils
from tools import _init_paths
from model.config import cfg
from data_augmentation.utils import xml_utils


def bright_adjuest_images_and_save(bright_adjuest, data_dir):
    jpg_path = os.path.join(data_dir, "JPEGImages")
    annot_path = os.path.join(data_dir, "Annotations")

    for bright in bright_adjuest:
        # io_utils.mkdir(os.path.join( data_dir+'_ver_flipped','JPEGImages'))
        for i in os.listdir(jpg_path):
            a, b = os.path.splitext(i)
            if b == ".jpg":
                img_path = os.path.join(jpg_path, i)
                img = cv2.imread(img_path)

                xml_path = os.path.join(annot_path, a + ".xml")
                object_infos = xml_store.get_object_infos_from_xml(xml_path)
                show_object_cv_box(object_infos, img)
                hor_flipped_img = bright_adjuest_image(img,bright)
                show_object_cv_box(object_infos, hor_flipped_img)

                break
                new_img_name = a + "_bright_adjuest_{}.jpg",format(bright)
                new_img_path = os.path.join(data_dir + "_bright_adjuest_{}",format(bright), 'JPEGImages', new_img_name)

                im_info = xml_utils.create_image_info(new_img_name, new_img_path, hor_flipped_img.shape[1],
                                                      hor_flipped_img.shape[0], hor_flipped_img.shape[2])
                new_xml_path = os.path.join(data_dir + "_bright_adjuest_{}",format(bright), 'Annotations')

                cv2.imwrite(new_img_path, hor_flipped_img)
                xml_store.save_annotations(new_xml_path, im_info, object_infos)

def bright_adjuest_image(im,gamma):
    from skimage import exposure
    im = exposure.adjust_gamma(im, gamma)
    return im


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--data_dir', dest='data_dir', help='prepare to compare this image and xml',
                        default=os.path.join(cfg.ROOT_DIR, 'data', 'train_data'))
    parser.add_argument('--package_dir', dest='package_dir', help='the compare data file name',
                        default="train_data-2018-03-07")
    parser.add_argument('--bright_adjuest', dest='bright_adjuest', help='how many the num you want to split',
                        default=[0.8,1.2])
    args = parser.parse_args()

    return args


args = parse_args()

if __name__ == "__main__":
    bright_adjuest_images_and_save(args.bright_adjuest, os.path.join(args.data_dir, args.package_dir))

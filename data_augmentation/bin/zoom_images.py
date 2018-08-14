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
                zoom_img = scaleit(img,scale)
                # object_infos_rotate=rotate_xml(object_infos,img,angle,rotated_img)
                show_object_cv_box(object_infos, zoom_img)
                break

                new_img_name=a + "_rotate_" + str(angle) + ".jpg"
                new_img_path=os.path.join( data_dir+'_zoom_{}'.format(angle),'JPEGImages', new_img_name)


                # print(object_infos)
                # show_object_cv_box(object_infos, rotated_img)
                # print(rotated_img.shape)
                im_info=xml_utils.create_image_info(new_img_name,new_img_path,rotated_img.shape[1],rotated_img.shape[0],rotated_img.shape[2])
                new_xml_path=os.path.join( data_dir+'_rotate_{}'.format(angle),'Annotations')


                cv2.imwrite(new_img_path, rotated_img)
                xml_store.save_annotations(new_xml_path, im_info, object_infos_rotate)

def Zoom(factor):  # 缩放
    # factor>1表示缩小；factor<1表示放大
    transform = np.array([[factor, 0, 0], [0, factor, 0], [0, 0, 1]])
    return transform

def scaleit(image, factor, isseg=False):
    order = 0 if isseg == True else 3
    height, width, depth= image.shape
    zheight = int(np.round(factor * height))
    zwidth = int(np.round(factor * width))
    zdepth = depth
    if factor < 1.0:
        newimg  = np.zeros_like(image)
        row = (height - zheight) // 2
        col = (width - zwidth) // 2
        layer = (depth - zdepth) // 2
        newimg[row:row+zheight, col:col+zwidth, layer:layer+zdepth] = interpolation.zoom(image, (float(factor), float(factor), 1.0), order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]
        return newimg
    elif factor > 1.0:
        row = (zheight - height) // 2
        col = (zwidth - width) // 2
        layer = (zdepth - depth) // 2
        newimg = interpolation.zoom(image[row:row+zheight, col:col+zwidth, layer:layer+zdepth], (float(factor), float(factor), 1.0), order=order, mode='nearest')
        extrah = (newimg.shape[0] - height) // 2
        extraw = (newimg.shape[1] - width) // 2
        extrad = (newimg.shape[2] - depth) // 2
        newimg = newimg[extrah:extrah+height, extraw:extraw+width, extrad:extrad+depth]
        return newimg
    else:
        return image

def rotate_xml(object_infos,img,angle,img_rotate):
    rotate_obj_infos=[]
    for object_info in object_infos:
        class_name,_,xmin, ymin, xmax, ymax = object_info.split(",")
        x_list=[int(xmin),int(xmax)]
        y_list=[int(ymin),int(ymax)]
        height, width = img.shape[0], img.shape[1]
        max_x,max_y=0,0
        min_x,min_y=width,height
        for x in x_list:
            for y in y_list:
                x1, y1 = rotate_point(width,height, angle, x, y)
                x1 =int(x1+(img_rotate.shape[1]/2))
                y1 =int(y1+(img_rotate.shape[0]/2))
                max_x = max([max_x, x1])
                min_x = min([min_x, x1])
                max_y = max([max_y, y1])
                min_y = min([min_y, y1])
        rotate_obj_infos.append("{},1,{},{},{},{}".format(class_name,min_x,min_y,max_x,max_y))
    return rotate_obj_infos


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--data_dir', dest='data_dir', help='prepare to compare this image and xml',
                        default=os.path.join(cfg.ROOT_DIR,'data','train_data'))
    parser.add_argument('--package_dir', dest='package_dir', help='the compare data file name',
                        default="train_data-2018-03-07")
    parser.add_argument('--zoom_scale', dest='zoom_scale', help='how many the num you want to split',
                        default=[0.5,1.5])
    args = parser.parse_args()

    return args

args = parse_args()


if __name__=="__main__":
    zoom_images_and_save(args.zoom_scale,os.path.join(args.data_dir,args.package_dir))

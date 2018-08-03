import os
import argparse
import cv2
from data_augmentation.utils.xml_utils import show_object_cv_box
from data_augmentation.utils import xml_store,io_utils
from math import *
from tools import _init_paths
from model.config import cfg
from data_augmentation.utils import xml_utils
def rotate_images_and_save(angles,data_dir):
    jpg_path=os.path.join(data_dir,"JPEGImages")
    annot_path=os.path.join(data_dir,"Annotations")

    for angle in angles:
        io_utils.mkdir(os.path.join( data_dir+'_rotate_{}'.format(angle),'JPEGImages'))
        print(angle)
        for i in os.listdir(jpg_path):
            print(i)
            a, b = os.path.splitext(i)
            if b == ".jpg":
                img_path = os.path.join(jpg_path, i)
                img = cv2.imread(img_path)

                xml_path = os.path.join(annot_path, a + ".xml")
                object_infos = xml_store.get_object_infos_from_xml(xml_path)
                # show_object_cv_box(object_infos, img)
                rotated_img = rotate_image(img, angle)
                object_infos_rotate=rotate_xml(object_infos,img,angle,rotated_img)
                show_object_cv_box(object_infos_rotate, rotated_img)
                break

                new_img_name=a + "_rotate_" + str(angle) + ".jpg"
                new_img_path=os.path.join( data_dir+'_rotate_{}'.format(angle),'JPEGImages', new_img_name)


                # print(object_infos)
                # show_object_cv_box(object_infos, rotated_img)
                # print(rotated_img.shape)
                im_info=xml_utils.create_image_info(new_img_name,new_img_path,rotated_img.shape[1],rotated_img.shape[0],rotated_img.shape[2])
                new_xml_path=os.path.join( data_dir+'_rotate_{}'.format(angle),'Annotations')


                cv2.imwrite(new_img_path, rotated_img)
                xml_store.save_annotations(new_xml_path, im_info, object_infos_rotate)

def rotate_image(img,angle):
    img=img[:, ::-1, :]
    return img

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
# def rotate_point(width, angle, x, y):
#     x1 = fabs(sin(radians(angle))) * int(y) + fabs(cos(radians(angle))) * int(x)
#     y1 = fabs(sin(radians(angle))) * (width - int(x)) + fabs(cos(radians(angle))) * int(y)
#     return int(x1), int(y1)
def rotate_point(width,height,angle,x,y):
    x1=width-x
    y1=height-y
    return int(x1),int(y1)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--data_dir', dest='data_dir', help='prepare to compare this image and xml',
                        default=os.path.join(cfg.ROOT_DIR,'data','train_data'))
    parser.add_argument('--package_dir', dest='package_dir', help='the compare data file name',
                        default="train_data-2018-03-07")
    parser.add_argument('--angles', dest='angles', help='how many the num you want to split',
                        default=[1])
    args = parser.parse_args()

    return args

args = parse_args()


if __name__=="__main__":
    rotate_images_and_save(args.angles,os.path.join(args.data_dir,args.package_dir))

from PIL import Image
import os
import _init_paths
from model.config import cfg
import argparse
from data_processing.utils import io_utils
import matplotlib.pyplot as plt
import numpy as np

def crop(img,crop_size):
    img=img.crop(crop_size)
    return img

def load_image_files(data_dir):
    jpg_files=os.path.join(data_dir,"JPEGImages")
    image_files=[]
    for jpg in os.listdir(jpg_files):
        image_files.append(os.path.join(jpg_files,jpg))
    crop_jpg_files=os.path.join(data_dir,"JPEGImages_crop")
    crop_xml_files=os.path.join(data_dir,"Annotations_crop")
    io_utils.mkdir(crop_jpg_files)
    io_utils.mkdir(crop_xml_files)
    return image_files,crop_jpg_files,crop_xml_files

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--data_dir', dest='data_dir', help='prepare to compare this image and xml',
                        default=os.path.join(cfg.ROOT_DIR,"data","train_data"))
    parser.add_argument('--package_dir', dest='package_dir', help='the compare data file name',
                        default="train_data-2018-03-07")
    parser.add_argument('--size', dest='size', help='the compare data file name',
                        default=(800,800))
    args = parser.parse_args()

    return args

args = parse_args()

def create_crop_bbox(img_size,crop_size):
    crop_bbox=np.zeros((5,4),dtype=int)
    crop_bbox[0]=np.array((0,0,crop_size[0],crop_size[1]))
    crop_bbox[1]=np.array((0,img_size[1]-crop_size[1],crop_size[0],img_size[1]))
    crop_bbox[2]=np.array((img_size[0]-crop_size[0],0,img_size[0],crop_size[1]))
    crop_bbox[3]=np.array((img_size[0]-crop_size[0],img_size[1]-crop_size[1],img_size[0],img_size[1]))
    crop_bbox[4]=np.array((int(img_size[0]/2-crop_size[0]/2),int(img_size[1]/2-crop_size[1]/2),
                           int(img_size[0]/2+crop_size[0]/2),int(img_size[1]/2+crop_size[1]/2)))
    return crop_bbox

def crop_images(data_dir,crop_size):
    image_files, crop_jpg_files, crop_xml_files=load_image_files(data_dir)
    for image_file in image_files:
        image=Image.open(image_file)
        img_size=image.size
        # count=161
        if img_size[0]>crop_size[0] and img_size[1]>crop_size[1]:
            crop_bbox=create_crop_bbox(img_size,crop_size)
            plt.figure("crop image",figsize=(8,8))
            for box in crop_bbox:
                img=crop(image,box)

        break



        break


if __name__=="__main__":
    data_dir=os.path.join(args.data_dir,args.package_dir)
    crop_images(data_dir,args.size)


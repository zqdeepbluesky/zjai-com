import os
import random
import numpy
from data_processing.utils import io_utils
import argparse
import _init_paths
from model.config import cfg
import math
from zjai_createData.zjai_2_create_main import create_package_main

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--data_dir', dest='data_dir', help='prepare to compare this image and xml',
                        default=os.path.join(cfg.ROOT_DIR,"data","train_data"))
    parser.add_argument('--package_dir', dest='package_dir', help='the compare data file name',
                        default="train_data-2018-03-07")
    parser.add_argument('--split_num', dest='split_num', help='how many the num you want to split',
                        default=5)
    args = parser.parse_args()

    return args

args = parse_args()

def load_image_files(data_dir):
    image_path=os.path.join(data_dir,"JPEGImages")
    image_files=[]
    for image in os.listdir(image_path):
        image_files.append(os.path.join(image_path,image))
    return image_files

def get_random_index(num):
    index=list(range(0,num,1))
    random_index=index.copy()
    random.shuffle(random_index)
    return random_index

def split_files(data_dir,image_files,index,split_num):
    group_size=math.ceil(len(image_files)/split_num)
    print(index)
    split_file_lists=[]
    for i in range(split_num):
        start_num=i*group_size
        end_num=(i+1)*group_size
        group_index=[]
        if i+1==split_num:
            group_index = index[start_num:]
        else:
            group_index=index[start_num:end_num]
        for file_num in group_index:
            image_path=image_files[file_num]
            new_image_file=os.path.join(data_dir+"_sub{}".format(i+1),"JPEGImages")
            io_utils.mkdir(new_image_file)
            io_utils.copy(image_path,new_image_file)
            xml_path=image_path.replace("JPEGImages","Annotations").replace(".jpg",'.xml')
            new_xml_file=os.path.join(data_dir+"_sub{}".format(i+1),"Annotations")
            io_utils.mkdir(new_xml_file)
            io_utils.copy(xml_path, new_xml_file)
        split_file_lists.append(data_dir+"_sub{}".format(i+1))
    return split_file_lists




if __name__=="__main__":
    data_dir=os.path.join(args.data_dir,args.package_dir)
    image_files=load_image_files(data_dir)
    index=get_random_index(len(image_files))
    split_file_lists=split_files(data_dir,image_files,index,args.split_num)
    scale=9
    for split_file in split_file_lists:
        create_package_main(split_file,scale)
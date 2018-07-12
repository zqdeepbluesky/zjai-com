# -*- coding: utf-8 -*-
# @Time    : 5/21/2018
# @Author  : CarrieChen
# @File    : six2one.py
# @Software: ZJ_AI
# this code is for paste six images into one background.
# another way:put the same label of objects into a queue,and pick up six labels each time.the key point is random and cycle.


from PIL import Image
import utils.io_utils
import os
import random
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import utils.fusion_utils
import cv2

def handle_background(input_path, min_side, max_side,output_path):
    ncount = 17982
    utils.io_utils.mkdir(output_path)
    utils.io_utils.remove_all(output_path)
    for f in os.listdir(input_path):
        img_path = os.path.join(input_path, f)
        img = cv2.imread(img_path)
        (rows, cols, _) = img.shape
        smallest_side = min(rows, cols)
        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side
        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        # resize the image with the computed scale
        img = cv2.resize(img, None, fx=scale, fy=scale)
        temp=output_path
        output_path = os.path.join(output_path,"bg" + "_" + str_date[0:4] + "-" + str_date[4:6] + "-" + str_date[6:8] + "-" + str(ncount) + ".jpg")
        cv2.imwrite(output_path, img)  # output img name,change here!!
        # return img, scale
        ncount += 1
        output_path=temp
    return output_path

#read one background each time
def read_background(input_path,n):
    background_list=os.listdir(input_path)
    background = os.path.join(input_path,background_list[n])
    return background

#copy from src folder and rename each img and put the result in rename folder
def rename(src_folder,rename_folder):
    utils.io_utils.mkdir(rename_folder)
    utils.io_utils.remove_all(rename_folder)
    for f in os.listdir(src_folder):
        rename_sub_folder=os.path.join(rename_folder,f)
        utils.io_utils.mkdir(rename_sub_folder)
        src_sub_folder=os.path.join(src_folder,f)
        utils.io_utils.copy_dir(src_sub_folder,rename_sub_folder)
    for e in os.listdir(rename_folder):
        for a in os.listdir(os.path.join(rename_folder,e)):
            try:
                os.rename(os.path.join(rename_folder,e,a),os.path.join(rename_folder,e,e+"-"+a))
            except FileExistsError:
                pass
    return rename_folder

#get the catalog of a folder
def read_object(rename_path):
    label_list=os.listdir(rename_path)
    catalog={}
    for f in label_list:
        img_list=os.listdir(os.path.join(rename_path,f))
        catalog[f]=img_list
    return catalog

#random get six imgs from all imgs catalog and save the choosed catalog into a dict
def get_six_imgs(count):
    result={}
    keys=list(count.keys())   #py3 need list()
    keys_slice=random.sample(keys,6)
    for s in keys_slice:
        values=count[s]
        values_slice=random.sample(values,1)
        result[s]=values_slice[0]
    return result

#use the functions in fusion_utils.py
def handle_img(img,part):
    scale_rate=0.5+random.random()
    #rotate_angle = 360 * random.random()
    rotate_angle = random.sample([0,90,180,270],1)[0]
    transpose_mode = random.sample(['H','V'],1)
    brightness_rate = 0.5 + random.random()
    constrast_rate = 0.5+ random.random()
    color_rate = 0.5+random.random()
    sharpness_rate =0.5+random.random()
    img=utils.fusion_utils.scale(img,scale_rate)    #change the scale_rate
    img=utils.fusion_utils.rotate(img,rotate_angle)
    img=utils.fusion_utils.transpose(img,transpose_mode)
    img=utils.fusion_utils.enhance_bright(img,brightness_rate)
    img=utils.fusion_utils.enhance_constrast(img,constrast_rate)
    img=utils.fusion_utils.enhance_color(img,color_rate)
    img=utils.fusion_utils.enhance_sharpness(img,sharpness_rate)
    img= utils.fusion_utils.resize(img, part)
    return img

#paste six images into a background
def paste_images(input_path,bg,six_imgs_dict,output_path,nimgs):
    utils.io_utils.mkdir(output_path)
    background=Image.open(bg)
    object_path_list=[]
    xml_info={}
    for k in list(six_imgs_dict.keys()):
        object_path=os.path.join(input_path,k,six_imgs_dict[k])
        object_path_list.append(object_path)
    width,height=background.size
    part=(int(width/3),int(height/2))
    nwidth,nheight=0,0
    for o in object_path_list:
        object_path=o
        object=Image.open(object_path)
        object=handle_img(object,part)
        box = [nwidth * part[0] + 50, nheight * part[1] + 50]  # don't keep to the side
        background.paste(object, box)
        split_list = o.split("\\")  # change here if linux system
        label_name = split_list[split_list.index('Rename') + 1]
        xml_info[label_name]=box
        nwidth=nwidth+1
        if(nwidth%3==0):
            nheight=nheight+1
            nwidth=0
        if(nheight%2==0):
            nheight=0
    save_path=os.path.join(output_path,"fusion"+"_"+str_date[0:4]+'-'+str_date[4:6]+'-'+str_date[6:8]+"_"+str(nimgs)+".jpg")
    background.save(save_path)
    create_xml(xml_path,"fusion"+"_"+str_date[0:4]+'-'+str_date[4:6]+'-'+str_date[6:8]+"_"+str(nimgs),save_path,object.size,xml_info)
    nimgs=nimgs+1
    return nimgs

#create xml
def create_xml(output_path,filename,path,size,obj_info):
    utils.io_utils.mkdir(output_path)
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = filename  + ".jpg"

    node_path = SubElement(node_root, 'path')
    node_path.text = path

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(size[0])

    node_height = SubElement(node_size, 'height')
    node_height.text = str(size[1])

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(3)

    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'

    for k in list(obj_info.keys()):

        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        caption = "{}".format(k)
        node_name.text = caption

        node_pose = SubElement(node_object, 'pose')
        node_pose.text = 'Unspecified'

        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'

        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        #check_border(b, Image_info.im_info.width, Image_info.im_info.height)

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(obj_info[k][0])

        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(obj_info[k][1])

        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(obj_info[k][2])

        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(obj_info[k][3])

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)
    xml_path = os.path.join(output_path, filename + '.xml')
    print(xml_path)
    with open(xml_path, 'w+') as f:
        dom.writexml(f, addindent='', newl='', encoding='utf-8')


#some paths
parent_path='F:\\bg' #change there
src_path=os.path.join(parent_path,"Src") #should be prepare before
rename_path=os.path.join(parent_path,"Rename")
background_path=os.path.join(parent_path,"Background") #should be prepare before
background_origin_path=os.path.join(background_path,"origin2")#should be prepare before
background_handle_path=os.path.join(background_path,"after2")
after_paste_path=os.path.join(parent_path,"JPEGImages")
xml_path=os.path.join(parent_path,"Annotations")
str_date = '{year}{month}{day}'.format(year='2018', month='05', day='30')  #改这里，日期

if __name__ == "__main__":
    background_handle_path=handle_background(background_origin_path, 800, 1333, background_handle_path)
    # nbackground=0
    # nimgs=10000  #count all imgs and this is the start num
    # all_labels = {}  #count all labels
    # rename_path= rename(src_path,rename_path)
    # all_labels=read_object(rename_path)
    # bg_count = len(os.listdir(background_handle_path))
    # loop = 5   #change loop
    # while loop>0:
    #     while nbackground < bg_count:
    #         six_imgs_dict = get_six_imgs(all_labels)
    #         background = read_background(background_handle_path, nbackground)
    #         nimgs=paste_images(rename_path,background,six_imgs_dict,after_paste_path,nimgs)
    #         nbackground += 1
    #     loop-=1
    #     nbackground=0  #important

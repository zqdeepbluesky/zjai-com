from PIL import Image
import os
import _init_paths
from model.config import cfg
import argparse
from data_processing.utils import io_utils
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
from morelib.utils import xml_fromsg

def crop(img,crop_size):
    img=img.crop(crop_size)
    return img

def load_image_files(data_dir):
    jpg_files=os.path.join(data_dir,"JPEGImages")
    image_files=[]
    for jpg in os.listdir(jpg_files):
        image_files.append(os.path.join(jpg_files,jpg))
    crop_jpg_files=os.path.join(data_dir+"_crop","JPEGImages")
    crop_xml_files=os.path.join(data_dir+"_crop","Annotations")
    io_utils.mkdir(data_dir+"_crop")
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
    parser.add_argument('--size', dest='size', help='the crop size',
                        default=(800,800))
    parser.add_argument('--scale', dest='scale', help='the resize scale',
                        default=1000)
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

def get_info_from_xml(xml_path):
    et = ET.parse(xml_path)
    element = et.getroot()
    element_objs = element.findall('object')
    obj_info = []
    for element_obj in element_objs:
        node = element_obj.find('name')
        label = node.text
        bbox = element_obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)
        obj_info.append("{},1,{},{},{},{}".format(label,x1,y1,x2,y2))
    return obj_info

def save_data_into_xml(image,size,xml_path,result_data):
    im_info = {"path": image}
    im_info["width"] = size[0]
    im_info["height"] = size[1]
    im_info["name"] = os.path.splitext(os.path.split(image)[1])[0]
    im_info["channel"] = size[2]
    xml_fromsg.save_annotations(xml_path, im_info, result_data)

def compare_obj_info(obj_info,crop_box):
    new_obj_info=[]
    for obj in obj_info:
        x1,y1,x2,y2 = obj.split(",")[2:]
        if (int(x1)>=crop_box[0] and int(x1)<=crop_box[2]) or (int(x2)>=crop_box[0] and int(x2)<=crop_box[2]):
            if (int(y1)>=crop_box[1] and int(y1)<=crop_box[3]) or (int(y2)>=crop_box[1] and int(y2)<=crop_box[3]):
                x1=min(max(int(x1),crop_box[0]),crop_box[2])-crop_box[0]+1
                y1 = min(max(int(y1), crop_box[1]), crop_box[3])-crop_box[1]+1
                x2 = min(max(int(x2), crop_box[0]), crop_box[2])-crop_box[0]-1
                y2 = min(max(int(y2), crop_box[1]), crop_box[3])-crop_box[1]-1
                new_obj_info.append("{},1,{},{},{},{}".format(obj.split(",")[0],x1,y1,x2,y2))
    return new_obj_info

def show_crop_image(image,obj_info):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, aspect='equal')
    for i in range(len(obj_info)):
        classes,_,x1,y1,x2,y2=obj_info[i].split(",")
        bbox=[int(x1),int(y1),int(x2),int(y2)]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s}'.format(classes),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def get_min_box(obj_info):
    xmin,ymin,xmax,ymax=obj_info[0].split(",")[2:]
    xmin, ymin, xmax, ymax=int(xmin),int(ymin),int(xmax),int(ymax)
    for obj in obj_info[1:]:
        x1, y1, x2, y2 = obj.split(",")[2:]
        xmin= min(int(xmin),int(x1))
        ymin = min(int(ymin), int(y1))
        xmax = max(int(xmax), int(x2))
        ymax = max(int(ymax), int(y2))
    for i in range(len(obj_info)):
        classes, score, x1, y1, x2, y2 = obj_info[i].split(",")
        obj_info[i] = "{},1,{},{},{},{}".format(classes, int(x1)-xmin, int(y1)-ymin,int(x2)-xmin, int(y2)-ymin)
    return [xmin,ymin,xmax,ymax],obj_info

def resize_box(obj_info,scala):
    for i in range(len(obj_info)):
        classes,score,x1,y1,x2,y2=obj_info[i].split(",")
        obj_info[i]="{},1,{},{},{},{}".format(classes,int(int(x1)/scala),int(int(y1)/scala),int(int(x2)/scala),int(int(y2)/scala))
    return obj_info

def cal_scale(image,scale):
    height=image.size[1]
    scale= height/scale
    return scale


def crop_images(data_dir,crop_size):
    image_files, crop_jpg_files, crop_xml_files=load_image_files(data_dir)
    for image_file in image_files:
        print(image_file)
        xml_path = image_file.replace("JPEGImages", "Annotations").replace(".jpg", ".xml")
        image=Image.open(image_file)
        scale = cal_scale(image,args.scala)
        if scale<=1:
            continue
        obj_info = get_info_from_xml(xml_path)
        image=image.resize((int(image.size[0]/scale),int(image.size[1]/scale)))
        obj_info = resize_box(obj_info, scale)
        # min_box,obj_info=get_min_box(obj_info)
        # image=crop(image,min_box)
        # plt.imshow(image)
        # plt.show()
        img_size = image.size
        # count=161
        if img_size[0]>crop_size[0] and img_size[1]>crop_size[1]:
            crop_bbox=create_crop_bbox(img_size,crop_size)
            count=1
            for box in crop_bbox:
                img=crop(image,box)
                new_obj_info=compare_obj_info(obj_info, box)
                if len(new_obj_info)==0:
                    continue
                show_crop_image(img, new_obj_info)
                filename=os.path.splitext(os.path.split(image_file)[1])[0]
                # xml_crop_path=os.path.join(crop_xml_files,filename+"_{}.xml".format(count))
                image_crop_path=os.path.join(crop_jpg_files,filename+"_{}.jpg".format(count))
                img.save(image_crop_path)
                save_data_into_xml(image_crop_path,[img.size[0],img.size[1],3],crop_xml_files,new_obj_info)
                count+=1
        else:

            # new_obj_info = compare_obj_info(obj_info, min_box)
            show_crop_image(image, new_obj_info)
            filename = os.path.splitext(os.path.split(image_file)[1])[0]
            # xml_crop_path=os.path.join(crop_xml_files,filename+"_{}.xml".format(count))
            image_crop_path = os.path.join(crop_jpg_files, filename + "_0.jpg")
            image.save(image_crop_path)
            save_data_into_xml(image_crop_path, [image.size[0], image.size[1], 3], crop_xml_files, new_obj_info)




if __name__=="__main__":
    data_dir=os.path.join(args.data_dir,args.package_dir)
    crop_images(data_dir,args.size)


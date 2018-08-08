import os
import os.path as osp
import xml.etree.ElementTree as ET
from morelib.utils import io_utils
import shutil

def check_bnds(root_dir,dataDirs):
    mainPath=osp.join(dataDirs,"ImageSets","Main")
    fileList=[]
    with open(mainPath+"/trainval.txt","r") as f:
        lineList=f.readlines()
        for line in lineList:
            line=line.replace("\n","")
            fileList.append(dataDirs+"/"+line+".xml")
    count=0
    for xml in fileList:
        check_xml(xml)
        count+=1
        if count%5000==0:
            print(count)

def check_xml(xml):
    et = ET.parse(xml.replace("JPEGImages","Annotations"))
    element = et.getroot()
    element_objs = element.findall('object')
    for element_obj in element_objs:
        bbox = element_obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        x2 = float(bbox.find('xmax').text)
        name = element_obj.find('name').text
        if x1>=x2:
            print(name,xml)

def check_xml_obj_exist(xml):
    et = ET.parse(xml.replace("JPEGImages","Annotations"))
    element = et.getroot()
    element_objs = element.findall('object')
    return len(element_objs)

def move(src_file, dest_dir):
    try:
        shutil.move(src_file, dest_dir)
        print("move successfuly:'{}' to '{}'".format(src_file, dest_dir))
    except Exception as e:
        print("Can't not move '{}' to '{}'. :{}", src_file, dest_dir, e)

def check_xml_obj_exists(data_dir):
    jpeg_valid=os.path.join(data_dir,"JPEGImages_valid")
    io_utils.mkdir(jpeg_valid)
    annot_path=os.path.join(data_dir,"Annotations")
    jpeg_path=os.path.join(data_dir,"JPEGImages")
    for xml in os.listdir(annot_path):
        xml_path=os.path.join(annot_path,xml)
        obj_num=check_xml_obj_exist(xml_path)
        if obj_num==0:
            move(os.path.join(jpeg_path,os.path.splitext(xml)[0]+".jpg"),os.path.join(jpeg_valid,os.path.splitext(xml)[0]+".jpg"))


if __name__=="__main__":
    root_dir = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    dataDirs = osp.join(root_dir, 'data', 'train_data',"predict_data-2018-07-24")
    # check_bnds(root_dir,dataDirs)
    check_xml_obj_exists(dataDirs)

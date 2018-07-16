import os
import os.path as osp
import xml.etree.ElementTree as ET

def checkAll(root_dir,dataDirs):
    mainPath=osp.join(dataDirs,"ImageSets","Main")
    fileList=[]
    with open(mainPath+"/trainval.txt","r") as f:
        lineList=f.readlines()
        for line in lineList:
            line=line.replace("\n","")
            fileList.append(dataDirs+"/"+line+".xml")
    count=0
    for xml in fileList:
        checkXml(xml)
        count+=1
        if count%5000==0:
            print(count)

def checkXml(xml):
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

if __name__=="__main__":
    # root_dir = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    root_dir="/home/hyl/data/ljk/project/2-shopDetect/tf-faster-rcnn-master"
    dataDirs = osp.join(root_dir, 'data', 'train_data')
    checkAll(root_dir,dataDirs)

import os
import os.path as osp
from data_processing.utils.io_utils import *

def copy_file(srcDirs,distDirs,fileList):
    for line in fileList:
        line=line.replace("\n","")
        jpg_file=osp.join(srcDirs,line+'.jpg')
        xml_file=osp.join(srcDirs,line+'.xml').replace("JPEGImages","Annotations")
        mkdir(osp.abspath(osp.dirname(osp.join(distDirs,line+".jpg"))))
        mkdir(osp.abspath(osp.dirname(osp.join(distDirs, line + ".xml").replace("JPEGImages","Annotations"))))
        copy(jpg_file,osp.join(distDirs,line+".jpg"))
        copy(xml_file,osp.join(distDirs, line + ".xml").replace("JPEGImages","Annotations"))
    print("total copy {} jpg and xml file".format(len(fileList)))

def get_txt_data(srcDirs,start_num,step,fileType):
    txtFile=osp.join(srcDirs,"ImageSets","Main","{}.txt".format(fileType))
    fileList=[]
    with open(txtFile,'r') as f:
        lineList=f.readlines()
        assert (start_num+step<len(lineList))
        for i in range(start_num,start_num+step):
            fileList.append(lineList[i].replace("\n",""))
    return fileList

def write_txt_file(distDirs,fileList):
    mkdir(osp.join(distDirs, "ImageSets", "Main"))
    txtFile = osp.join(distDirs, "ImageSets", "Main", "trainval.txt")
    with open(txtFile,'w') as f:
        f.write("\n".join(fileList))
    print("have wrote {} line".format(len(fileList)))




if __name__=="__main__":
    root_dir=osp.abspath(osp.join(osp.dirname(__file__),".."))
    root_dir="/home/hyl/data/ljk/project/2-shopDetect/tf-faster-rcnn-master/"
    srcDirs=osp.join(root_dir,"data","train_data")
    distDirs=osp.join(root_dir,"data","test_data")
    fileList=get_txt_data(srcDirs,23380,100,"test")
    write_txt_file(distDirs,fileList)
    copy_file(srcDirs,distDirs,fileList)

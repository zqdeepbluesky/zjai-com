import matplotlib.pyplot as plt
import math
from PIL import Image
import numpy as np
import os.path as osp
from ..utils.io_utils import *

def _create_Main(dataDirs,fileList,scale):
    '''
    create the trainval.txt and test.txt for train.
    trainval data : test data = 5:1
    :param path:
    :return:
    '''
    trainval_images = []
    test_images = []
    mkdir(osp.join(dataDirs,"ImageSets","Main"))
    for i in range(len(fileList)//scale, len(fileList)):
        s = fileList[i]
        if dataDirs[-1]=="/":
            s=s.replace(dataDirs,"")
        else:
            s=s.replace(dataDirs + "/", "")
        trainval_images.append(s.split('.')[0] + '\n')

    for i in range(len(fileList)//scale):
        s = fileList[i]
        if dataDirs[-1]=="/":
            s=s.replace(dataDirs,"")
        else:
            s=s.replace(dataDirs + "/", "")
        test_images.append(s.split('.')[0] + '\n')

    with open(dataDirs+'/ImageSets/Main/trainval.txt','w+') as f:
        f.writelines(trainval_images)
        print("{}, numbers:{}".format(dataDirs + '/trainval.txt', len(trainval_images)))
    with open(dataDirs+'/ImageSets/Main/test.txt','w+') as f:
        f.writelines(test_images)
        print("{}, numbers:{}".format(dataDirs + '/test.txt', len(test_images)))

    print('total: {}'.format(len(fileList)))
    print('step: {}'.format(len(trainval_images)//2+1))

def _create_Main_new(dataDirs,fileList,scale):
    '''
    create the trainval.txt and test.txt for train.
    trainval data : test data = 5:1
    :param path:
    :return:
    '''
    trainval_images = []
    test_images = []
    fileDict={}
    mkdir(osp.join(dataDirs,"ImageSets","Main"))
    for i in range(len(fileList)//scale, len(fileList)):
        s = fileList[i]
        parentDir=osp.dirname(s)
        filename=osp.splitext(s)[0].replace(parentDir,"").replace("/","")
        trainval_images.append(filename + '\n')
        fileDict[filename]=parentDir

    for i in range(len(fileList)//scale):
        s = fileList[i]
        parentDir = osp.dirname(s)
        filename = osp.splitext(s)[0].replace(parentDir, "").replace("/", "")
        test_images.append(filename + '\n')
        fileDict[filename] = parentDir

    with open(dataDirs+'/ImageSets/Main/trainval.txt','w+') as f:
        f.writelines(trainval_images)
        print("{}, numbers:{}".format(dataDirs + '/trainval.txt', len(trainval_images)))
    with open(dataDirs+'/ImageSets/Main/test.txt','w+') as f:
        f.writelines(test_images)
        print("{}, numbers:{}".format(dataDirs + '/test.txt', len(test_images)))
    with open(dataDirs+'/ImageSets/Main/filedict.txt','w+') as f:
        for key in fileDict.keys():
            f.write("{}|{}\n".format(key,fileDict[key]))
    print('total: {}'.format(len(fileList)))
    print('step: {}'.format(len(trainval_images)//2+1))

def create_package_main(data_dir,scale):
    fileList = get_all_file(data_dir, fileType="jpg")
    _create_Main_new(data_dir, fileList, scale)

def show_object_PIL_box(datas,img):
    plot_num=math.ceil((len(datas)+1)/3)*100+4*10+1
    img=np.ndarray(img)
    plt.figure("Image")
    plt.axis("off")
    plt.subplot(plot_num)
    plt.imshow(img), plt.axis('off')
    for data in datas:
        plot_num += 1
        box=data.split(",")
        x1 = int(box[2])
        y1 = int(box[3])
        x2 = int(box[4])
        y2 = int(box[5])
        plt.subplot(plot_num)
        plt.imshow(img.crop((x1, y1, x2, y2))), plt.axis('off')
    plt.show()

def show_object_cv_box(datas,img):
    plot_num=math.ceil((len(datas)+1)/3)*100+4*10+1
    plt.figure("Image")
    plt.axis("off")
    plt.subplot(plot_num)
    plt.imshow(img), plt.axis('off')
    for data in datas:
        plot_num += 1
        box=data.split(",")
        x1 = int(box[2])
        y1 = int(box[3])
        x2 = int(box[4])
        y2 = int(box[5])
        plt.subplot(plot_num)
        plt.imshow(img[y1:y2, x1:x2]), plt.axis('off')
    plt.show()


def get_all_file(dirs,fileType):
    fileList=[]
    for parent, dirnames, filenames in os.walk(dirs):
        for filename in filenames:
            fileName = os.path.join(parent, filename)
            if fileName[-3:] == fileType:
                fileList.append(fileName)
    return fileList

def compare(jpgList,xmlList):
    for jpgFile in jpgList:
        checkFile=jpgFile.replace("JPEGImages","Annotations").replace(".jpg",".xml")
        if checkFile  not in xmlList:
            print("{} jpg file is not have {} xml file".format(jpgFile,checkFile))
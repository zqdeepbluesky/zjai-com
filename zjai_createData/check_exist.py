import os
import os.path as osp
import sys

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


if __name__=="__main__":
    root_dir=osp.abspath(osp.join(osp.dirname(__file__), '..'))
    dataDirs=osp.join(root_dir,'data','train_data')
    print(dataDirs)
    jpgFileList=get_all_file(dataDirs,"jpg")
    xmlFileList=get_all_file(dataDirs,"xml")
    print("jpg files have {} files".format(len(jpgFileList)))
    print("xml files have {} files".format(len(xmlFileList)))
    compare(jpgFileList,xmlFileList)



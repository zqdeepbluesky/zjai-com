import os
import os.path as osp
from  xml.etree import ElementTree
import PIL.Image
from xml.etree.ElementTree import ElementTree
from  data_processing.utils.io_utils import mkdir

def get_xml(data_dirs):
    xmlList=[]
    for parent,dirname,filenames in os.walk(data_dirs):
        for filename in filenames:
            filepath=osp.join(parent,filename)
            if osp.splitext(filepath)[1]=='.xml':
                xmlList.append(filepath)
    return xmlList

def overwrite(xmlList):
    count=0
    for xmlpath in xmlList:
        jpgpath=xmlpath.replace("Annotations","JPEGImages").replace(".xml",".jpg")
        width,height=get_size(jpgpath)
        element=get_xml_data(xmlpath)
        element_objs = element.findall('size')
        for element_obj in element_objs:
            node = element_obj.find('width')
            node.text=str(int(width))
            node1 = element_obj.find('height')
            node1.text = str(int(height))
        save(xmlpath,element)
        if count%10000==0:
            print(count)
        count+=1
def save(xmlpath,tree):
    # xml = tostring(element, pretty_print=True)
    mkdir(osp.dirname(xmlpath.replace("Annotations","Annotations1")))
    tree.write(xmlpath.replace("Annotations","Annotations1"),encoding="utf-8",xml_declaration=True)



def get_size(jpgpath):
    sizes=PIL.Image.open(jpgpath).size
    return sizes[0],sizes[1]

def get_xml_data(xmlpath):
    tree = ElementTree()
    tree.parse(xmlpath)
    return tree

if __name__=="__main__":
    # root_dir = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    root_dir = "/home/hyl/data/ljk/project/2-shopDetect/tf-faster-rcnn-master/"
    dataDirs = osp.join(root_dir, 'data', 'train_data')
    xmlList=get_xml(dataDirs)
    overwrite(xmlList)
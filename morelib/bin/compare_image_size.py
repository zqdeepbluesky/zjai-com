import os
import os.path as osp
from  xml.etree import ElementTree
import PIL.Image
import argparse
import _init_paths
from model.config import cfg
from xml.etree.ElementTree import ElementTree
from  data_processing.utils.io_utils import mkdir

def _get_xml(data_dirs):
    xmlList=[]
    annot_path=os.path.join(data_dirs,"Annotations")
    for xml in os.listdir(annot_path):
        xmlList.append(os.path.join(annot_path,xml))
    return xmlList

def _overwrite(xmlList):
    count=0
    error_log=[]
    for xmlpath in xmlList:
        jpgpath=xmlpath.replace("Annotations","JPEGImages").replace(".xml",".jpg")
        width,height=_get_size(jpgpath)
        element=_get_xml_data(xmlpath)
        element_objs = element.findall('size')
        for element_obj in element_objs:
            node = element_obj.find('width')
            f_width=node.text
            node.text=str(int(width))
            node1 = element_obj.find('height')
            f_height=node1.text
            node1.text = str(int(height))
            if str(int(float(f_width)))!=str(int(width)) or str(int(float(f_height)))!=str(int(height)):
                error_log.append("{} size error,image is {},{} size,but xml is {},{}!".format(xmlpath,str(int(width)),str(int(height)),str(int(float(f_width))),str(int(float(height)))))
        _save(xmlpath,element)
        if count%10000==0:
            print(count)
        count+=1
    log_file=os.path.abspath(os.path.join(xmlList[0],"..","..","compare_log.txt"))
    with open(log_file,"w") as f:
        f.write("{} jpg hava {} error size, and hava {} right size\n\n".format(len(xmlList),len(error_log),len(xmlList)-len(error_log)))
        f.write("\n".join(error_log))

def _save(xmlpath,tree):
    # xml = tostring(element, pretty_print=True)
    mkdir(osp.dirname(xmlpath.replace("Annotations","Annotations_new")))
    tree.write(xmlpath.replace("Annotations","Annotations_new"),encoding="utf-8",xml_declaration=True)



def _get_size(jpgpath):
    sizes=PIL.Image.open(jpgpath).size
    return sizes[0],sizes[1]

def _get_xml_data(xmlpath):
    tree = ElementTree()
    tree.parse(xmlpath)
    return tree

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--data_dir', dest='data_dir', help='prepare to compare this image and xml',
                        default=os.path.join(cfg.ROOT_DIR,"data","train_data"))
    parser.add_argument('--package_dir', dest='package_dir', help='the compare data file name',
                        default="test_data-2018-07-20")
    args = parser.parse_args()

    return args

def _rename_file(data_dir):
    origin_file=os.path.join(data_dir,"Annotations")
    new_file=os.path.join(data_dir,"Annotations_new")
    os.rename(origin_file,os.path.join(data_dir,"Annotations_old"))
    os.rename(new_file, os.path.join(data_dir, "Annotations"))

args = parse_args()

if __name__=="__main__":
    data_set=os.path.join(args.data_dir,args.package_dir)
    xmlList=_get_xml(data_set)
    _overwrite(xmlList)
    _rename_file(data_set)
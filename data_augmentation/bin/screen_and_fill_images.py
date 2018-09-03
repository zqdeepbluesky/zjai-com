import os
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from data_augmentation.utils import io_utils,xml_store
import xml.etree.ElementTree as ET
import cv2
import numpy as np
def get_label(xml_path):
    et = ET.parse(xml_path)
    element = et.getroot()
    element_objs = element.findall('object')
    labels=[]
    for i,element_obj in enumerate(element_objs):
        node = element_obj.find('name').text
        labels.append(node)
    return labels

def get_object_infos_from_xml(xml_path,target_labels):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    element_objs = root.findall('object')
    object_infos=[]
    labels=[]
    for element_obj in element_objs:
        name = element_obj.find('name').text
        if name in target_labels:
            bbox = element_obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            labels.append(name)
            object_infos.append("{},1,{},{},{},{}".format(name,xmin,ymin,xmax,ymax))
    return object_infos,labels

def get_object_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    element_objs = root.findall('object')
    object_infos=[]
    labels=[]
    for element_obj in element_objs:
        name = element_obj.find('name').text
        bbox = element_obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        labels.append(name)
        object_infos.append("{},1,{},{},{},{}".format(name,xmin,ymin,xmax,ymax))
    return object_infos,labels

def save_data_into_xml(image,im,xml_path,result_data):
    im_info = {"path": image}
    im_info["width"] = im.shape[1]
    im_info["height"] = im.shape[0]
    im_info["name"] = os.path.splitext(os.path.split(image)[1])[0]+".jpg"
    im_info["channel"] = im.shape[2]
    xml_store.save_annotations(xml_path, im_info, result_data)

def screen_image(im,target_labels,result_data):
    labels=[x.split(',')[0] for x in result_data]
    boxes=[x.split(',')[2:] for x in result_data]
    fill_data=[]
    store_dara=[]
    for i,label in enumerate(labels):
        if label not in target_labels:
            fill_data.append(result_data[i])
        else:
            store_dara.append(result_data[i])
    return fill_data,store_dara

def fill_image(im,fill_data):
    screen_im=im.copy()
    boxes = [x.split(',')[2:] for x in fill_data]
    mask=im.copy()
    for i,box in enumerate(boxes):
        xmin=int(boxes[i][0])
        ymin=int(boxes[i][1])
        xmax=int(boxes[i][2])
        ymax=int(boxes[i][3])
        screen_im[ymin:ymax,xmin:xmax,:]=0
        mask[ymin:ymax,xmin:xmax,:]=255
    return mask




if __name__=="__main__":
    src_path='/home/hyl/data/ljk/github-pro/zjai-com/data/train_data/all_train_data_resize2'
    dist_path='/home/hyl/data/ljk/github-pro/zjai-com/data/train_data/train_data-2018-08-29_c'
    # io_utils.mkdir(dist_path)
    # io_utils.mkdir(os.path.join(dist_path,'JPEGImages'))
    # io_utils.mkdir(os.path.join(dist_path,'Annotations'))
    # target_labels=['hn-hnwssgnyl-gz-yw-250ml','hly-hlydhp-hz-yw-138g','lzs-rnbdwhbg-hz-nlw-145g',
    #                'ls-lswxsp-tz-yw-104g','hly-hlyytdst-dz-ljnpw-70g','wt-wtnmc-pz-nm-500ml']
    target_labels=['shj-shjxxp-dz-xxw-40g','hly-hlytlmsp-hz-yw-138g','xpp-xppnc-bz-yw-80g','wwsp-wznn-hz-yw-125ml',
                    'wl-wldmj-dz-lw-106g','hly-hlyytdst-dz-scjyljw-45g','fd-fdtsyl-gz-yw-330ml','jdb-jdb-pz-yw-500ml',
                   'ch-cht-pz-ygw-30g','kkkl-jcnmwqs-pz-nmw-500ml','wwsp-wzxmt-dz-nnw-210g','wwsp-wznn-gz-yw-145ml',
                    'md-mdwssyl-pz-qn-600ml','wwsp-wwxb-dz-yw-84g','sh-shjrhtc-dz-jrw-240g','llm-llm-dz-nmcm-60g',
                   'yd-ydwtkxt-pz-xnmgw-56g','wwsp-wwxxs-dz-yw-60g','yj-pjfz-dz-sjw-100g','ys-zzyspyz-hz-yw-245ml']
    target_labels = ['shj-shjxxp-dz-xxw-40g', 'hly-hlytlmsp-hz-yw-138g', 'xpp-xppnc-bz-yw-80g', 'wwsp-wznn-hz-yw-125ml',
                     'wl-wldmj-dz-lw-106g', 'hly-hlyytdst-dz-scjyljw-45g', 'fd-fdtsyl-gz-yw-330ml',
                     'jdb-jdb-pz-yw-500ml','ch-cht-pz-ygw-30g', 'kkkl-jcnmwqs-pz-nmw-500ml']
    loss_labels=['wwsp-wzxmt-dz-nnw-210g','wwsp-wznn-gz-yw-145ml','md-mdwssyl-pz-qn-600ml','wwsp-wwxb-dz-yw-84g',
                 'sh-shjrhtc-dz-jrw-240g','llm-llm-dz-nmcm-60g','yd-ydwtkxt-pz-xnmgw-56g','wwsp-wwxxs-dz-yw-60g',
                 'yj-pjfz-dz-sjw-100g','ys-zzyspyz-hz-yw-245ml']
    count=[0 for i in range(len(target_labels))]
    num=-1
    for xml in os.listdir(os.path.join(src_path,"Annotations")):
        num += 1
        xml_path=os.path.join(src_path,"Annotations",xml)
        result_data,labels=get_object_from_xml(xml_path)
        index=-1
        # result_data=set_other(loss_labels,result_data)
        if len(result_data)>0 :
            for label in labels:
                if label in target_labels:
                    index=target_labels.index(label)
                    break
            if index!=-1 and count[index]<=20000:
                count[index]+=1
                im=cv2.imread(os.path.join(src_path,"JPEGImages",xml[:-3]+"jpg"))
                fill_data, store_dara=screen_image(im, target_labels, result_data)
                im=fill_image(im,fill_data)
                cv2.imshow('test',im)
                cv2.waitKey(0)
                image=xml[:-3]+"jpg"
                break
                # save_data_into_xml(image, im, os.path.join(dist_path,'Annotations'), result_data)
                # io_utils.copy(os.path.join(src_path,"JPEGImages",xml[:-3]+"jpg"), os.path.join(dist_path, "JPEGImages"))
            # if sum(count)==2000:
            #     break
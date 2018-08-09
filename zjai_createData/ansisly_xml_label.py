import xml.etree.ElementTree as ET
import os
from morelib.utils import io_utils

def get_xml_speci_label(xmlPath,label_list):
    '''
    函数用于得到xml文件的object信息
    :param xmlPath:
    :return:
    '''
    if os.path.exists(xmlPath)!=1:
        print(xmlPath)
    et = ET.parse(xmlPath)
    element = et.getroot()
    element_objs = element.findall('object')
    labelList=[]
    for element_obj in element_objs:
        node = element_obj.find('name')
        label=node.text
        if label in label_list:
            labelList.append(label)
    return len(labelList),labelList

def compare_from_xml(xmlPath1,xmlPath2,label_list,save_path):
    io_utils.mkdir(save_path)
    xmlFileList1=[]
    xmlFileList2 = []
    for xmlFile in os.listdir(xmlPath1):
        xmlFileList1.append(os.path.join(xmlPath1,xmlFile))
        xmlFileList2.append(os.path.join(xmlPath2, xmlFile))

    print(len(xmlFileList1),len(xmlFileList2))
    tp_sum=0
    fp_sum=0
    fn_sum=0
    d_sum=0
    t_sum=0
    for i in range(len(xmlFileList1)):
        tp=0
        fp=0
        fn=0
        xmlFile1=xmlFileList1[i]
        xmlFile2=xmlFileList2[i]
        d_labelNum, d_labelList=get_xml_speci_label(xmlFile1,label_list)
        t_labelNum, t_labelList=get_xml_speci_label(xmlFile2,label_list)
        for d_label in d_labelList:
            if d_label in t_labelList:
                labenIndex=t_labelList.index(d_label)
                t_labelList.remove(t_labelList[labenIndex])
                tp+=1
            else:
                fp+=1
            fn=t_labelNum-tp
        tp_sum+=tp
        fp_sum+=fp
        fn_sum+=fn
        if t_labelNum!=0:
            io_utils.copy(xmlFile1.replace("Annotations_test","JPEGImages").replace(".xml",".jpg"),save_path)
            io_utils.copy(xmlFile1,save_path)
        d_sum+=d_labelNum
        t_sum+=t_labelNum
        # print(xmlFile1,xmlFile2,tp,fp,fn,d_labelNum,t_labelNum)
    prec=tp_sum/(fp_sum+tp_sum)
    recall=tp_sum/(tp_sum+fn_sum)
    print(prec,recall)
    print(tp_sum,fp_sum,fn_sum,d_sum,t_sum)
    return "{},{},{},{},{},{},{}".format(prec,recall,tp_sum,fp_sum,fn_sum,d_sum,t_sum)

if __name__=="__main__":
    xmlPath1="/home/hyl/data/ljk/github-pro/zjai-com/data/train_data/random_choice_data_3000/Annotations_test"
    xmlPath2="/home/hyl/data/ljk/github-pro/zjai-com/data/train_data/random_choice_data_3000/Annotations"
    save_path='/home/hyl/data/ljk/github-pro/zjai-com/data/predict_data/tm_images'
    target_label=['nfsq-nfsqyytrs-pz-yw-550ml','ty-hzy-pz-gw-500ml','yb-ybcjs-pz-yw-555ml']
    compare_from_xml(xmlPath1,xmlPath2,target_label,save_path)
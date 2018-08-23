import xml.etree.ElementTree as ET
import os
import prettytable as pt
import numpy as np

def add_label_dict(xmlPath,label_dict):
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
    for element_obj in element_objs:
        node = element_obj.find('name')
        label=node.text
        if label in label_dict.keys():
            label_dict[label]+=1
        else:
            label_dict[label] = 1
    return label_dict

def get_xml_label_num(xmlPath):
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
    count=len(element_objs)
    labelList=[]
    for element_obj in element_objs:
        node = element_obj.find('name')
        label=node.text
        labelList.append(label)
    return count,labelList

def get_tabs(test_infos):
    tb = pt.PrettyTable()
    tb.field_names = ["model_name","test_data",'label','presion','recall',"detect_num", "actual_num", "tp_num", "fp_num",'fn_num','ap']
    for test_info in test_infos:
        info=test_info.split(",")
        tb.add_row(info)
    return tb

def save_tb_in_images(path,tb):
    from PIL import Image, ImageDraw, ImageFont
    tab_info = str(tb)
    space = 5
    im = Image.new('RGB', (30, 30), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im, "RGB")
    img_size = draw.multiline_textsize(tab_info)
    im_new = im.resize((img_size[0] + space * 2, img_size[1] + space * 2))
    del draw
    del im
    draw = ImageDraw.Draw(im_new, 'RGB')
    draw.multiline_text((space, space), tab_info, fill=(255, 255, 255))
    im_new.save(path+".png", "PNG")
    del draw

def save_tb_in_txt(path,tb):
    # tb.field_names = ["模型名称", "测试数据", '精确率', '召回率', "模型识别总数", "实际总数", "正确识别数量", "误识别总数",
    #                   '漏识别总数']
    f = open(path+'.txt', "a+")
    f.write(str(tb))
    f.write('\n')
    f.close()

def save_tb_in_xml(path,tb):
    f = open(path+'.xml', "a+")
    s = tb.get_html_string()
    f.write(str(s))
    f.close()

def get_xml_field_name(path):
    et = ET.parse(path)
    element = et.getroot()
    element_objs = element.findall('tr')
    field_name=[]
    for ele in element_objs:
        td=ele.findall('th')
        for t in td:
            field_name.append(t.text)
    return field_name

def get_xml_row_info(path):
    et = ET.parse(path)
    element = et.getroot()
    element_objs = element.findall('tr')
    row_info=[]
    for ele in element_objs:
        td=ele.findall('td')
        for t in td:
            row_info.append(t.text)
    return row_info

def merge_tb_from_xml(path_list):
    count =0
    tb = pt.PrettyTable()
    for path in path_list:
        if count==0:
            field_name=get_xml_field_name(path)
            tb.field_names=field_name
            row_info=get_xml_row_info(path)
            tb.add_row(row_info)
        else:
            row_info = get_xml_row_info(path)
            tb.add_row(row_info)
        count+=1
    print(tb)

def summary_tb(tb,test_infos):
    presion,recall,d_num,t_num,tp_num,fp_num,fn_num,count=0,0,0,0,0,0,0,0
    model_name,test_data='','total'
    for test_info in test_infos:
        infos = test_info.split(",")
        if count == 0:
            model_name = infos[0]
        count+=1
        d_num+=int(infos[5])
        t_num+=int(infos[6])
        tp_num+=int(infos[7])
        fp_num+=int(infos[8])
        fn_num+=int(infos[9])
    presion=tp_num/(tp_num+fp_num)
    recall=tp_num/(tp_num+fn_num)
    tb.add_row([model_name,test_data,'total',presion,recall,d_num,t_num,tp_num,fp_num,fn_num,'//'])
    return tb





def cal_model_acc(xmlPath1,xmlPath2,cal_label=False):
    xmlFileList1 = []
    xmlFileList2 = []
    for xmlFile in os.listdir(xmlPath1):
        xmlFileList1.append(os.path.join(xmlPath1, xmlFile))
        xmlFileList2.append(os.path.join(xmlPath2, xmlFile))

    tp_sum,fp_sum,fn_sum,d_sum,t_sum = 0,0,0,0,0
    for i in range(len(xmlFileList1)):
        tp,fp,fn = 0,0,0
        xmlFile1 = xmlFileList1[i]
        xmlFile2 = xmlFileList2[i]
        d_labelNum, d_labelList = get_xml_label_num(xmlFile1)
        t_labelNum, t_labelList = get_xml_label_num(xmlFile2)
        for d_label in d_labelList:
            if d_label in t_labelList:
                labenIndex = t_labelList.index(d_label)
                t_labelList.remove(t_labelList[labenIndex])
                tp += 1
            else:
                fp += 1
            fn = t_labelNum - tp
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
        # if fp !=0 or fn !=0:
        #     io_utils.copy(xmlFile1.replace("Annotations","JPEGImages").replace(".xml",".jpg"),save_path)
        #     io_utils.copy(xmlFile1,save_path)
        d_sum += d_labelNum
        t_sum += t_labelNum
        # print(xmlFile1,xmlFile2,tp,fp,fn,d_labelNum,t_labelNum)
    # print(tp_sum, fp_sum, fn_sum, d_sum, t_sum)
    prec = tp_sum / (fp_sum + tp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    # print(prec, recall)
    return "{},{},{},{},{},{},{}".format(prec, recall,d_sum, t_sum, tp_sum, fp_sum, fn_sum)
def get_xml_label_bnd(xmlPath):
    if os.path.exists(xmlPath)!=1:
        print(xmlPath)
    et = ET.parse(xmlPath)
    element = et.getroot()
    element_objs = element.findall('object')

    labelList=[]
    boxes=np.zeros((1,4),dtype=int)
    for i,element_obj in enumerate(element_objs):
        node = element_obj.find('name')
        label=node.text
        labelList.append(label)
        bbox = element_obj.find('bndbox')
        if i==0:
            boxes[0,:]=np.array([int(bbox.find('xmin').text),int(bbox.find('ymin').text),int(bbox.find('xmax').text),int(bbox.find('ymax').text)])
        else:
            box = np.array([int(bbox.find('xmin').text), int(bbox.find('ymin').text), int(bbox.find('xmax').text),
                             int(bbox.find('ymax').text)])
            boxes=np.row_stack((boxes,box))
    return labelList,boxes

def init_ind(class_name):
    tp_sum = np.zeros(class_name, dtype=int)
    fp_sum = np.zeros(class_name, dtype=int)
    fn_sum = np.zeros(class_name, dtype=int)
    d_sum = np.zeros(class_name, dtype=int)
    t_sum = np.zeros(class_name, dtype=int)
    prec = np.zeros(class_name, dtype=float)
    rec = np.zeros(class_name, dtype=float)
    return tp_sum, fp_sum, fn_sum, d_sum, t_sum,prec,rec

def cal_iou(gt_boxes,box):
    ixmin = np.maximum(gt_boxes[:, 0], box[0])
    iymin = np.maximum(gt_boxes[:, 1], box[1])
    ixmax = np.minimum(gt_boxes[:, 2], box[2])
    iymax = np.minimum(gt_boxes[:, 3], box[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((box[2] - box[0] + 1.) * (box[3] - box[1] + 1.) +
           (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.) *
           (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.) - inters)

    overlaps = inters / uni
    ovmax = np.max(overlaps)  # 重叠度IOU最大的
    jmax = np.argmax(overlaps)
    return ovmax,jmax

def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
def cal_label_acc(xmlPath1,xmlPath2,CLASSES):
    classes=list(CLASSES)
    xmlFileList1=[os.path.join(xmlPath1, xmlFile) for xmlFile in os.listdir(xmlPath1)]
    xmlFileList2 = [os.path.join(xmlPath2, xmlFile) for xmlFile in os.listdir(xmlPath1)]
    tp_arr, fp_arr, fn_arr, d_sum_arr, t_sum_arr,prec_arr,rec_arr = init_ind(len(classes))
    for i in range(len(xmlFileList1)):
        xmlFile1 = xmlFileList1[i]
        xmlFile2 = xmlFileList2[i]
        d_labelList,d_boxes = get_xml_label_bnd(xmlFile1)
        t_labelList, t_boxes = get_xml_label_bnd(xmlFile2)
        for t_label in t_labelList:
            class_index = classes.index(t_label)
            t_sum_arr[class_index] += 1
        for j,d_label in enumerate(d_labelList):
            class_index=classes.index(d_label)
            d_sum_arr[class_index]+=1
            if d_label in t_labelList:
                tp_arr[class_index]+=1
                t_labelList.remove(d_label)

    fp_arr=d_sum_arr-tp_arr
    fn_arr=t_sum_arr-tp_arr
    prec_arr=tp_arr/(tp_arr+fp_arr)
    prec_arr[np.isnan(prec_arr)]=0
    rec_arr =tp_arr/(tp_arr+fn_arr)
    rec_arr[np.isnan(rec_arr)] = 0
    return ["{},{},{},{},{},{},{}".format(prec_arr[num], rec_arr[num],d_sum_arr[num], t_sum_arr[num],
                                          tp_arr[num], fp_arr[num], fn_arr[num]) for num in range(len(classes))]

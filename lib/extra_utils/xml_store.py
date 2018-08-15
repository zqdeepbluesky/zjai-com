from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import os
from lib.extra_utils import io_utils
import xml.etree.ElementTree as ET

def make_xml(im_info, datas):
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text =im_info['name']

    node_path = SubElement(node_root, 'path')
    node_path.text = im_info["path"]

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text =str(im_info["width"])

    node_height = SubElement(node_size, 'height')
    node_height.text =str(im_info["height"])

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(im_info["channel"])

    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'

    node_root=set_object(datas,node_root,im_info)

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)

    return dom

def set_object(datas,node_root,im_info):
    for data in datas:
        label=data.split(",")[0]
        b=list(map(int,data.split(",")[2:]))
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = label

        node_pose = SubElement(node_object, 'pose')
        node_pose.text = 'Unspecified'

        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'

        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        check_border(b, im_info["width"],im_info["height"])

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(b[0]))

        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(b[1]))

        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(b[2]))

        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(b[3]))
    return node_root

def check_border(bbox, width, height):
    if len(bbox) <4:
        return

    if bbox[0] <= 0.0:
        bbox[0]= 1

    if bbox[1] <= 0.0:
        bbox[1] = 1

    if bbox[2] >= width:
        bbox[2] = width - 1

    if bbox[3] >= height:
        bbox[3] = height - 1

def _beautifulFormat(xmlDomObject):
    if xmlDomObject:
        xmlStr = xmlDomObject.toprettyxml(indent='', newl='', encoding='utf-8')
        xmlStr = xmlStr.replace(b'\t', b'').replace(b'\n', b'')
        xmlDomObject = parseString(xmlStr)
        xmlStr = xmlDomObject.toprettyxml(indent='\t', newl='\n', encoding='utf-8')
        dom = parseString(xmlStr)
        return dom
    else:
        return False

def save_annotations(save_dir, im_info, data):
    dom = make_xml(im_info, data)
    io_utils.mkdir(save_dir)
    xml_path = os.path.join(save_dir, im_info["name"].replace(".jpg",".xml"))
    with open(xml_path, 'w+') as f:
        dom.writexml(f, addindent='', newl='', encoding='utf-8')

def save_data_into_xml(image,im,xml_path,result_data):
    im_info = {"path": image}
    print(im.shape)
    im_info["width"] = im.shape[1]
    im_info["height"] = im.shape[0]
    im_info["name"] = os.path.splitext(os.path.split(image)[1])[0]+".jpg"
    im_info["channel"] = im.shape[2]
    save_annotations(xml_path, im_info, result_data)

def get_object_infos_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    element_objs = root.findall('object')
    object_infos=[]
    for element_obj in element_objs:
        name = element_obj.find('name').text
        bbox = element_obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        object_infos.append("{},1,{},{},{},{}".format(name,xmin,ymin,xmax,ymax))
    return object_infos


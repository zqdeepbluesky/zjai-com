import os
from xml.dom.minidom import parseString


def make_xml(im_info, classDict):
    from lxml.etree import Element, SubElement, tostring
    from xml.dom.minidom import parseString
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

    for label in classDict:
        b=classDict[label]
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

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)

    return dom

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
def save_annotations(save_dir, im_info, classDict):
    dom = make_xml(im_info, classDict)
    xml_path = os.path.join(save_dir, im_info["name"].replace(".jpg",".xml"))
    with open(xml_path, 'w+') as f:
        dom.writexml(f, addindent='', newl='', encoding='utf-8')
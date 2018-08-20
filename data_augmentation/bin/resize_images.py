import cv2
import os
from morelib.utils import io_utils,xml_store
from data_augmentation.utils import xml_utils
from data_augmentation.utils.xml_utils import show_object_cv_box
from data_augmentation.utils import xml_store,io_utils

def get_images(image_dir):
    images_path=[]
    image_dir=os.path.join(image_dir,"JPEGImages")
    for image in os.listdir(image_dir):
        images_path.append(os.path.join(image_dir,image))
    return images_path

def cal_scale(size,min_b=1200):
    scale=min(size[0],size[1])/min_b
    return scale

def resize_images(images_path,image_dir):
    new_data_dir=image_dir+'_resize'
    io_utils.mkdir(new_data_dir)
    new_jpg_path=image_dir + '_resize'+'/JPEGImages'
    io_utils.mkdir(new_jpg_path)
    new_annot_path=image_dir + '_resize'+'/Annotations'
    io_utils.mkdir(new_annot_path)
    for image in images_path:
        a=os.path.split(image)[1][:-4]
        print(a,image)
        im=cv2.imread(image)
        scale=cal_scale(im.shape)
        resize_im=cv2.resize(im,None,fx=(1/scale),fy=(1/scale))
        xml_path = os.path.join(image_dir,'Annotations', a + ".xml")
        object_infos = xml_store.get_object_infos_from_xml(xml_path)
        new_object_infos=resize_box(object_infos, scale)
        # show_object_cv_box(new_object_infos, resize_im)
        new_image_path=os.path.join(image_dir+"_resize",'JPEGImages',os.path.split(image)[1][:-4]+"_resize.jpg")

        print(new_image_path)
        cv2.imwrite(new_image_path,resize_im)
        im_info = xml_utils.create_image_info(os.path.split(new_image_path)[1], os.path.split(new_image_path)[0], resize_im.shape[1], resize_im.shape[0],
                                              resize_im.shape[2])
        new_xml_path = new_annot_path
        xml_store.save_annotations(new_xml_path, im_info, new_object_infos)
def resize_box(obj_info,scala):
    for i in range(len(obj_info)):
        classes,score,x1,y1,x2,y2=obj_info[i].split(",")
        obj_info[i]="{},1,{},{},{},{}".format(classes,int(int(x1)/scala),int(int(y1)/scala),int(int(x2)/scala),int(int(y2)/scala))
    return obj_info
if __name__=="__main__":
    image_dir='/home/hyl/data/ljk/github-pro/zjai-com/data/train_data/train_data-2018-08-15'
    images_path=get_images(image_dir)
    resize_images(images_path,image_dir)
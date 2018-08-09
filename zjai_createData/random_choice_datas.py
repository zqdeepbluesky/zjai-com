import random
from morelib.utils import io_utils
import os

def get_packages_data(package_dir):
    choice_num=3000
    images_list=[]
    xml_list=[]
    for package in package_dir:
        jpg_path=os.path.join(package,'JPEGImages')
        xml_path=os.path.join(package,"Annotations")
        for image in os.listdir(jpg_path):
            images_list.append(os.path.join(jpg_path,image))
            xml_list.append(os.path.join(xml_path,image[:-4]+".xml"))
    index=[i for i in range(len(images_list))]
    random.shuffle(index)
    new_package='/home/hyl/data/ljk/github-pro/zjai-com/data/train_data/random_choice_data_{}'.format(choice_num)
    new_jpg_path=os.path.join(new_package,"JPEGImages")
    new_xml_path=os.path.join(new_package,"Annotations")
    io_utils.mkdir(new_package)
    io_utils.mkdir(new_jpg_path)
    io_utils.mkdir(new_xml_path)
    for i in index[:choice_num]:
        io_utils.copy(images_list[i],new_jpg_path)
        io_utils.copy(xml_list[i],new_xml_path)



if __name__=="__main__":
    package_dir=['/home/hyl/data/ljk/github-pro/zjai-com/data/train_data/all_train_data_resize2']
    get_packages_data(package_dir)

import PIL
from PIL import Image
from data_processing.utils import io_utils
import os
import cv2
import numpy as np
import PIL.Image
import multiprocessing
import multiprocessing.dummy

from multiprocessing import pool

def get_images_file(root_dir):
    image_files=[]
    for im_file in os.listdir(root_dir):
        image_files.append(os.path.join(root_dir,im_file))
    return image_files


def get_image_size(image_files):
    width_list=[]
    height_list=[]
    count=0
    for path in image_files:
        im_size=Image.open(path).size
        count+=1
        if count%5000==0:
            print(count)

        width_list.append(im_size[0])
        height_list.append(im_size[1])
    width_list=list(set(width_list))
    height_list = list(set(height_list))
    width_list.sort()
    height_list.sort()
    return width_list,height_list

def check_images_min(root_dir):
    image_files = get_images_file(root_dir)
    width_list, height_list = get_image_size(image_files)
    print(width_list[:100])
    print(height_list[:100])

def IsValidImage(path): #find empty file in a dir
    isValid = False
    num=0
    size=os.path.getsize(path)
    if size==0:
        print(path)
        isValid=True

    return isValid


def move_little_size(image_files,move_path):
    log_file=os.path.join(os.path.dirname(move_path),'check_log.txt')
    log_list=[]
    valid_num=0
    size_num=0
    io_utils.mkdir(move_path)
    for path in image_files:
        isValid = IsValidImage(path)
        if isValid:
            log_list.append("{} is valid,move!")
            valid_num+=1
            io_utils.move(path, move_path)
        else:
            im_size = PIL.Image.open(path).size
            min_size=min(im_size[0],im_size[1])
            if min_size<2000:
                size_num+=1
                io_utils.move(path,move_path)
                log_list.append("{} size({},{}) is less than 800,move!".format(os.path.split(path)[1],im_size[0],im_size[1]))
    print(valid_num,size_num)
    with open(log_file,'w+') as f:
        f.write('\n'.join(log_list))


def resize_image(img, min_side=800, max_side=1333):
    (rows, cols, _) = img.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale

def save_image(img, path):
    try:
        cv2.imwrite(path, img)
        # print(path)
    except Exception as ex:
        print(ex)

def read_image_rgb(path):
    try:
        image = np.asarray(PIL.Image.open(path).convert('RGB'))
    except Exception as ex:
        print('{}'.format(path))

    return image.copy()

# laod image
def load_image(path):
    return read_image_rgb(path)

def process_resize(image_path):
    image = load_image(image_path)
    img, scale =  resize_image(image)
    output_path = os.path.join(output,os.path.split(image_path)[1])
    # print(output_path)
    save_image(img, output_path)

def process_transfor(image_path):
    im = cv2.imread(image_path)
    name = os.path.split(image_path)[1]
    save_image(im[:, :, ::-1], os.path.join(output, name))

def process_choice(image_path):
    im = cv2.imread(image_path)
    if min(im.shape[0],im.shape[1])>=2000 and not IsValidImage(image_path):
        name = os.path.split(image_path)[1]
        save_image(im, os.path.join(output, name))


def main(root_dir,output):
    image_files = get_images_file(root_dir)
    count=0
    for image_path in image_files:
        process_choice(image_path)
        count+=1
        if count%5000==0:
            print(count)

def mutil_process(func):
    all_data = get_images_file(input)
    cpus = os.cpu_count()- 1
    p = multiprocessing.pool.Pool(cpus)
    p.map_async(func, all_data)

    p.close()
    p.join()


input = '/home/ljk/data/train_data/background'
output = '/home/ljk/data/train_data/background_min_2000'
io_utils.mkdir(output)

if __name__=="__main__":

    # # single process
    # # main(input,output)
    #
    #
    # # mutil process
    # # mutil_process()
    mutil_process(process_choice)


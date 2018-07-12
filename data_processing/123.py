
from PIL import Image
import os.path
import glob
import io_utils
import cv2

#resize and rename
def handle(img, min_side, max_side,output_path,f):
    img=cv2.imread(img)

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
    output_path=os.path.join(output_path,f)
    cv2.imwrite(output_path,img) #output img name,change here!!
    #return img, scale



#some path
parent_path="C:\\Users\\Administrator\\Desktop\\data_processing_carriechen\\count_all_annotations\\pic+lab166"
origin_path=os.path.join(parent_path,"origin")
handle_path=os.path.join(parent_path,"after")



if __name__  == '__main__':
    io_utils.mkdir(handle_path)
    io_utils.remove_all(handle_path)
    for f in os.listdir(origin_path):
        img_path=os.path.join(origin_path,f)
        handle(img_path,500,700,handle_path,f)  #background:800*1333


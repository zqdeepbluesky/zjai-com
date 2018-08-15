import numpy as np
from math import *
import cv2
from skimage import exposure


def _zoom_boxes(boxes, scale):
    new_boxes = np.zeros(boxes.shape)
    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        min_x = int(int(xmin) * scale[0])
        min_y = int(int(ymin) * scale[1])
        max_x = int(int(xmax) * scale[0])
        max_y = int(int(ymax) * scale[1])
        new_boxes[i] = [min_x, min_y, max_x, max_y]
    return new_boxes

def _shift_boxes(boxes,size,offset):
    def fix_new_key(key, offset, bound):
        if offset >= 0:
            key = min(key, bound)
        else:
            key = max(0, key)
        return key
    new_boxes = np.zeros(boxes.shape)
    box_count=0
    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax=boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
        xmin = fix_new_key(int(int(xmin) + offset[0]), offset[0], size[0])
        ymin = fix_new_key(int(int(ymin) + offset[1]), offset[1], size[1])
        xmax = fix_new_key(int(int(xmax) + offset[0]), offset[0], size[0])
        ymax = fix_new_key(int(int(ymax) + offset[1]), offset[1], size[1])
        if xmax - xmin != 0 and ymax - ymin != 0:
            new_boxes[box_count]=[xmin,ymin,xmax,ymax]
            box_count+=1
    return new_boxes[:box_count]

def _rotate_boxes(boxes,size,angle):
    def rotate_point(width, height, angle, x, y):
        x1 = (x - (width / 2)) * cos(radians(angle)) + (y - (height / 2)) * sin(radians(angle))
        y1 = (y - height / 2) * cos(radians(angle)) - (x - width / 2) * sin(radians(angle))
        return int(x1), int(y1)
    new_boxes=np.zeros(boxes.shape)
    height, width = size[1], size[0]
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))
    for i in range(boxes.shape[0]):
        x_list = [int(boxes[i][0]),int(boxes[i][2])]
        y_list = [int(boxes[i][1]),int(boxes[i][3])]
        max_x, max_y = 0, 0
        min_x, min_y = widthNew, heightNew
        for x in x_list:
            for y in y_list:
                x1, y1 = rotate_point(width, height, angle, x, y)
                x1 = int(x1 + (widthNew / 2))
                y1 = int(y1 + (heightNew / 2))
                max_x = max([max_x, x1])
                min_x = min([min_x, x1])
                max_y = max([max_y, y1])
                min_y = min([min_y, y1])
        new_boxes[i]=[int(min_x),int(min_y),int(max_x),int(max_y)]
    return new_boxes,widthNew,heightNew

def cal_resize_scale(image,scale):
    height=image.size[1]
    scale= height/scale
    return scale

def _bright_adjuest_image(im,gamma):
    im = exposure.adjust_gamma(im, gamma)
    return im

def _rotate_image(img,angle):
    height,width=img.shape[0],img.shape[1]
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步，目前不懂为什么加这步
    matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))
    return imgRotation

def _shift_image(image, offset, isseg=False):
    from scipy.ndimage.interpolation import shift
    order = 0
    return shift(image, (int(offset[0]), int(offset[1]), 0), order=order, mode='nearest')

def _zoom_image(image, factor_x,factor_y, isseg=False):
    from scipy.ndimage import interpolation
    order = 0 if isseg == True else 3
    newimg = interpolation.zoom(image, (float(factor_y), float(factor_x), 1.0), order=order, mode='nearest')
    return newimg

def get_crop_bbox(img_size,crop_size,position):
    crop_bbox=np.zeros((5,4),dtype=int)
    crop_bbox[0]=np.array((0,0,crop_size[0],crop_size[1]))
    crop_bbox[1]=np.array((0,img_size[1]-crop_size[1],crop_size[0],img_size[1]))
    crop_bbox[2]=np.array((img_size[0]-crop_size[0],0,img_size[0],crop_size[1]))
    crop_bbox[3]=np.array((img_size[0]-crop_size[0],img_size[1]-crop_size[1],img_size[0],img_size[1]))
    crop_bbox[4]=np.array((int(img_size[0]/2-crop_size[0]/2),int(img_size[1]/2-crop_size[1]/2),
                           int(img_size[0]/2+crop_size[0]/2),int(img_size[1]/2+crop_size[1]/2)))
    position_list=['lu','ld','ru','rd','m']
    position_index=position_list.index(position)
    return crop_bbox[position_index]

def create_crop_bbox(img_size,crop_size):
    crop_bbox=np.zeros((5,4),dtype=int)
    crop_bbox[0]=np.array((0,0,crop_size[0],crop_size[1]))
    crop_bbox[1]=np.array((0,img_size[1]-crop_size[1],crop_size[0],img_size[1]))
    crop_bbox[2]=np.array((img_size[0]-crop_size[0],0,img_size[0],crop_size[1]))
    crop_bbox[3]=np.array((img_size[0]-crop_size[0],img_size[1]-crop_size[1],img_size[0],img_size[1]))
    crop_bbox[4]=np.array((int(img_size[0]/2-crop_size[0]/2),int(img_size[1]/2-crop_size[1]/2),
                           int(img_size[0]/2+crop_size[0]/2),int(img_size[1]/2+crop_size[1]/2)))
    return crop_bbox

def _resize_box(boxes,scale):
    for i in range(len(boxes)):
        x1,y1,x2,y2=boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
        x1=int(x1/scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)
        boxes[i]=[x1,y1,x2,y2]
    return boxes


def crop(img,box):
    img=img[box[1]:box[3], box[0]:box[2]]
    return img

def cal_scale(size,scale):
    scale= min(size[0],size[1])/scale
    return scale

def random_crop_image(img, crop_size,scale,position):
    resize_scale = cal_scale(img.shape, scale)
    resize_image = cv2.resize(img,None,fx=1/resize_scale, fy=1/resize_scale)
    img_size=(resize_image.shape[1],resize_image.shape[0])
    crop_box = get_crop_bbox(img_size, crop_size,position)
    crop_img = crop(resize_image, crop_box)
    return crop_img


def cal_random_crop_box(boxes,crop_box):
    new_boxes = np.zeros(boxes.shape, dtype=int)
    count=0
    for i in range(len(boxes)):
        xmin,ymin,xmax,ymax = boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
        x1,y1,x2,y2=crop_box[0],crop_box[1],crop_box[2],crop_box[3]
        min_x =min(max(x1,int(xmin)),x2)-x1
        min_y = min(max(y1, int(ymin)), y2)-y1
        max_x = min(max(x1, int(xmax)), x2)-x1
        max_y = min(max(y1, int(ymax)), y2)-y1
        if max_x-min_x>0 and max_y-min_y>0:
            new_boxes[i]=[min_x,min_y,max_x,max_y]
            count += 1
    return new_boxes[:count]

def resize_box(boxes,scala):
    for i in range(len(boxes)):
        x1,y1,x2,y2=boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
        boxes[i]=[int(int(x1)/scala),int(int(y1)/scala),int(int(x2)/scala),int(int(y2)/scala)]
    return boxes
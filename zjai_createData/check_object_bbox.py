import matplotlib.pyplot as plt
import math
import numpy as np

def show_object_PIL_box(datas,img):
    plot_num=math.ceil((len(datas)+1)/3)*100+4*10+1
    img=np.ndarray(img)
    plt.figure("Image")
    plt.axis("off")
    plt.subplot(plot_num)
    plt.imshow(img), plt.axis('off')
    for data in datas:
        plot_num += 1
        box=data.split(",")
        x1 = int(box[2])
        y1 = int(box[3])
        x2 = int(box[4])
        y2 = int(box[5])
        plt.subplot(plot_num)
        plt.imshow(img.crop((x1, y1, x2, y2))), plt.axis('off')
    plt.show()

def show_object_cv_box(datas,img):
    plot_num=math.ceil((len(datas)+1)/3)*100+4*10+1
    plt.figure("Image")
    plt.axis("off")
    plt.subplot(plot_num)
    plt.imshow(img), plt.axis('off')
    for data in datas:
        plot_num += 1
        box=data.split(",")
        x1 = int(box[2])
        y1 = int(box[3])
        x2 = int(box[4])
        y2 = int(box[5])
        plt.subplot(plot_num)
        plt.imshow(img[y1:y2, x1:x2]), plt.axis('off')
    plt.show()


# im=blobs['data'][0,:,:,:]
# print(im)
# print(blobs['gt_boxes'],blobs['gt_boxes'].shape,im.shape)
# im_info=[]
# for i in range(blobs['gt_boxes'].shape[0]):
#     im_info.append('{},1,{},{},{},{}'.format(blobs['gt_boxes'][i,-1],int(blobs['gt_boxes'][i,0]),
#                                         int(blobs['gt_boxes'][i,1]),int(blobs['gt_boxes'][i,2]),
#                                         int(blobs['gt_boxes'][i, 3])))
# from zjai_createData.check_object_bbox import show_object_cv_box
# show_object_cv_box(im_info,im)
#
# import time as t
# t.sleep(1000)
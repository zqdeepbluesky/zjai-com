# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

"""
# # array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])
#  先以左上角(0,0)为例生成9个anchor，然后在向右向下移动，生成整个feature map所有点对应的anchor
"""

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.

    :param base_size: base_size = 16 也就是feature map上一点对应到原图的大小为16x16的区域
    :param ratios: ratios=[0.5,1,2] 指的是要将16x16的区域，按照1:2,1:1,2:1三种比例进行变换
    :param scales: 将输入的区域（16x16）的宽和高进行三种倍数，2^3=8，2^4=16，2^5=32倍的放大，如16x16的区域变成(16*8)*(16*8)=128*128的区域，(16*16)*(16*16)=256*256的区域，(16*32)*(16*32)=512*512的区域
    :return:
    """

    '''base_anchor值为[ 0,  0, 15, 15]: 表示最基本的一个大小为16x16的区域，四个值，分别代表这个区域的左下角和右下角的点的坐标 '''
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0])])

    """
    a=[1,2,3]
    b=[4,5,6]
    print(np.vstack((a,b)))
    
    [[1 2 3]
    [4 5 6]]
    """
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    输入的anchor的四个坐标值转化成（宽，高，中心点横坐标，中心点纵坐标）的形式
    :param anchor:存储了窗口左上角，右下角的坐标
    :return:返回width,height,(x,y)中心坐标,  将原来的anchor坐标（0，0，15，15）转化成了w:16,h:16,x_ctr=7.5,y_ctr=7.5的形式
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1

    # anchor中心点坐标
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    给定一组宽高向量，输出各个anchor，即预测窗口，**输出anchor的面积相等，只是宽高比不同

    :param ws: #ws:[23 16 11]，hs:[12 16 22],ws和hs一一对应
    :param hs: #ws:[23 16 11]，hs:[12 16 22],ws和hs一一对应
    :param x_ctr:
    :param y_ctr:
    :return:(x1,y1, x2,y2)
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))

    """
    a=[[1],[2],[3]]
    b=[[1],[2],[3]]
    c=[[1],[2],[3]]
    d=[[1],[2],[3]]
    print(np.hstack((a,b,c,d)))
    
    [[1 1 1 1]
     [2 2 2 2]
     [3 3 3 3]]
    """

    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.

    :param anchor: 一个anchor(四个坐标值表示)
    :param ratios: 三种宽高比例（0.5,1,2）
    :return:
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h  # size:16*16=256
    size_ratios = size / ratios  # 256/ratios[0.5,1,2]=[512,256,128]
    # round()方法返回x的四舍五入的数字，sqrt()方法返回数字x的平方根
    ws = np.round(np.sqrt(size_ratios))  # ws:[23 16 11]
    hs = np.round(ws * ratios)  # hs:[12 16 22],ws和hs一一对应。as:23&12

    # 给定一组宽高向量，输出各个预测窗口，也就是将（宽，高，中心点横坐标，中心点纵坐标）的形式，转成四个坐标值的形式
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)

    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    ratio_anchors中的三种宽高比的anchor，再分别进行三种scale的变换，也就是三种宽高比，搭配三种scale，最终会得到9种宽高比和scale 的anchors
    :param anchor:
    :param scales:
    :return:
    """

    # 枚举一个anchor的各种尺度，以anchor[0 0 15 15]为例,scales[8 16 32]
    w, h, x_ctr, y_ctr = _whctrs(anchor) #返回宽高和中心坐标，w:16,h:16,x_ctr:7.5,y_ctr:7.5
    ws = w * scales   #[128 256 512]
    hs = h * scales   #[128 256 512]
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr) #[[-56 -56 71 71] [-120 -120 135 135] [-248 -248 263 263]]

    return anchors

'''枚举一个anchor的各种尺度，以anchor[0 0 15 15]为例,scales[8 16 32]'''
if __name__ == '__main__':
    import time

    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed;

    embed()

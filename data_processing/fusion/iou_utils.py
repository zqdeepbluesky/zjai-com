# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 8/23/2018 3:07 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def intersection_rect(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])

    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return [x1,y1,x2,y2]

def intersection(a, b):
    i_rect = intersection_rect(a,b)
    i_area = area(i_rect)
    return i_area

def area(a):
    w = a[2] - a[0]
    h = a[3] - a[1]
    # print('area:', w * h)
    return  w * h

def union(a, b):
    area_a = area(a)
    area_b = area(b)
    area_i = intersection(a, b)
    return   area_a + area_b - area_i

def calc_iou(a, b):
    try:
        u = union(a, b)
        i = intersection(a, b)
        if abs(u)  < 0.000000001:
            return 1

        iou =  i / u
    except Exception as ex:
        print("{}，{},{}".format(i, u, ex))
    return iou

def test_calc_iou():
    # curr_iou = calc_iou([10,10,100,100], [50, 50, 200, 200])
    curr_iou = calc_iou([10,10,100,100], [10,10,100,100])
    return curr_iou

if __name__ == '__main__':
   print(test_calc_iou())
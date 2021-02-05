# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 09:40:29 2019

@author: QDW
"""

import numpy as np
"""计算构造laobel框和每个label框的重叠区域"""
def IoU(box, boxes):
#box：构造label框一个
#boxes:label框可能有多个。
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1) 
    area = (boxes[:, 2] - boxes[:, 0] + 1) * \
    (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    
    inter = w * h
    over = inter / (box_area + area - inter)
    return over
#box = np.array([1, 1, 4, 4])
#boxes = np.array([[2, 2, 5, 5], [3, 3, 6, 6]])
#print(IoU(box,boxes))
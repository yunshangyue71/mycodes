#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:56:25 2019
@author: q
"""
"""
box:自己构造的一个新的box
boxes:某张图片的所有boxes，
return：返回的是是一个list，返回的是这个构造的框和这个图片中
"""
import numpy as np
def IoU(box, boxes):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1) 
    boxes_area = (boxes[:, 2] - boxes[:, 0] + 1) *  (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    
    inter = w * h
    over = inter / (box_area + boxes_area - inter)
    return over

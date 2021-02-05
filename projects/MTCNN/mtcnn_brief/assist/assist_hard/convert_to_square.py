#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 06:43:44 2019

@author: q
"""
import numpy as np
def convert_to_square(box):
    square_box = box.copy()
    h = box[:, 3] - box[:, 1] + 1
    w = box[:, 2] - box[:, 0] + 1
    max_side = np.maximum(h,w)
    square_box[:, 0] = box[:, 0] + w*0.5 - max_side*0.5
    square_box[:, 1] = box[:, 1] + h*0.5 - max_side*0.5
    square_box[:, 2] = square_box[:, 0] + max_side - 1
    square_box[:, 3] = square_box[:, 1] + max_side - 1
    return square_box

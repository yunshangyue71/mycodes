#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 18:46:51 2019

@author: q
"""
import os
neg_path = '/media/q/deep/mtcnn_brief/pic_48_HHE/pic_48_neg_HHE'
pos_path = '/media/q/deep/mtcnn_brief/pic_48_HHE/pic_48_pos_HHE'
part_path = '/media/q/deep/mtcnn_brief/pic_48_HHE/pic_48_part_HHE'
neg_num = len(os.listdir(neg_path))
pos_num = len(os.listdir(pos_path))
part_num = len(os.listdir(part_path))
print(neg_num)
print(pos_num)
print(part_num)

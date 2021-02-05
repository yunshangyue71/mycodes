#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:48:36 2019
@author: QDW
neg比较多所以如果超过了25w*3,就只保留25w*3个
pos，part从里面可以重复的选取25w个
mark：全部保留
"""
import numpy as np
import os

from path_and_config import path

pic_size = 48#第一次PNet
#pic_size = 24#第二次RNet
#pic_size = 48#第三次ONet
choice = ['_E','_HE','_HHE']
if pic_size == 12: c = 0 
if pic_size == 24: c = 1
if pic_size == 48: c = 2
pnp_pic_hard_sample = choice[c]#训练数据的hard sample情况


pos_txt_dir = os.path.join(path.root,'pic_txt/pic_%d_pos%s.txt'%(pic_size,pnp_pic_hard_sample))
neg_txt_dir = os.path.join(path.root,'pic_txt/pic_%d_neg%s.txt'%(pic_size,pnp_pic_hard_sample))
part_txt_dir = os.path.join(path.root,'pic_txt/pic_%d_part%s.txt'%(pic_size,pnp_pic_hard_sample))
mark_txt_dir = os.path.join(path.root,'pic_txt/mark_%d.txt'%(pic_size))
out_txt_dir =   os.path.join(path.root,'pic_txt/pic_%d%s.txt'%(pic_size,pnp_pic_hard_sample))
with open(pos_txt_dir, 'r')as f:
    pos = f.readlines()
with open(neg_txt_dir, 'r')as f:
    neg = f.readlines()
with open(part_txt_dir, 'r')as f:
    part = f.readlines()
with open(mark_txt_dir, 'r')as f:
    mark = f.readlines()
    
with open(out_txt_dir, 'w') as f:
    nums = [len(neg), len(pos), len(part)]
    ratio = [3, 1, 1]
    base_num = 250000
    print('len(neg),len(pos),len(part),len(mark),base_num:\n',  len(neg), len(pos), len(part),len(mark), base_num)
    if len(neg) > base_num * 3:
        neg_keep = np.random.choice(len(neg), size = base_num*3,replace = True)
    else:
        neg_keep = np.random.choice(len(neg), size = len(neg),replace = True)     
    pos_keep = np.random.choice(len(pos), size = base_num, replace =True)
    part_keep = np.random.choice(len(part), size = base_num, replace =True)
    print('len(neg_keep), len(pos_keep), len(part_keep),len(mark)\n',
          len(neg_keep), len(pos_keep), len(part_keep),len(mark))
    
    
    for i in pos_keep:
        f.write(pos[i])
    for i in neg_keep:
        f.write(neg[i])
    for i in part_keep:
        f.write(part[i])
    for item in mark:
        f.write(item)
    
    




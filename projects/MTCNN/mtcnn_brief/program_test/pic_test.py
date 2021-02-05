#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 07:37:40 2019

@author: q
这个程序测试生成的图片的形状以及显示某一个
"""
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time
import numpy as np

from path_and_config import path

folder_dir = os.path.join(path.root,'mark_12')
pic_expect_size = (12,12,3)
count = 0
for pic_name in os.listdir(folder_dir):
    pic_dir = os.path.join(folder_dir,pic_name)
    pic = cv2.imread(pic_dir)
    pic_real_size = pic.shape
    if pic_real_size != pic_expect_size:
        
        print(pic_dir)
        print(pic_real_size)
        print(pic_expect_size)
        raise Exception
    if count % 200 == 0:
        img = Image.open(pic_dir)
        plt.imshow(img)
        plt.show()
        time.sleep(2)
    #print(count)
    count += 1
    



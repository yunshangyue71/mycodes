#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:16:39 2019

@author: q
"""
import os 
import random 
from os.path import join
import cv2
import numpy as np
import numpy.random as npr

from path_and_config import path
from assist.iou import IoU
from assist.mark_infoget_imgchange import rotate, flip,getDataFromTxt, BBox

"""准备工作"""
pic_size = 24#第一次PNet
#pic_size = 24#第二次RNet
#pic_size = 48#第三次ONet

expand = True#如果标记的图片数量少，可以通过旋转镜像等来增加mark图片数量
mark_txt = 'txt/trainImageList.txt'
txt_save_dir = os.path.join(path.root,'pic_txt')
mark_save_dir = os.path.join(path.root,'mark_%d'%pic_size)
if not os.path.exists(mark_save_dir):
    os.mkdir(mark_save_dir)
f = open(os.path.join(txt_save_dir, 'mark_%s.txt'%(pic_size)), 'w')
mark_data = getDataFromTxt(mark_txt, data_path = path.mark)
print('total data is ',len(mark_data))

"""txt中box区域图片生成，txt的生成，"""
img_done_count = 0#计数图片标记点，所有图片的所有label狂的信息提取出了多少
mark_pic_count = 0#最终符合条件的个数mark图像
for (imgPath, box, marks) in mark_data:
    a_img_a_box_pic_resized = []#一个照片输出的所有的mark_pic
    a_img_marks_rate = []#所有照片的所有标记点
    img = cv2.imread(imgPath)
    if img is None:
        print(imgPath,box,marks,img_done_count)
        
    #txt中的数据，这个是第一i部分
    img_h, img_w, img_c = img.shape
    gt_box = np.array([box.left, box.top, box.right, box.bottom])
    box_pic = img[box.top : box.bottom+1, box.left : box.right+1]
    box_pic = cv2.resize(box_pic, (pic_size, pic_size))
    marks_rate = np.zeros((5, 2))
    for index, a_mark in enumerate(marks):
        rv = ((a_mark[0] - gt_box[0]) / (gt_box[2] - gt_box[0]), 
              (a_mark[1] - gt_box[1]) / (gt_box[3] - gt_box[1]))
        marks_rate[index] = rv
    a_img_a_box_pic_resized.append(box_pic)
    a_img_marks_rate.append(marks_rate.reshape(10))
    #第二层，每个box随机构造一个box
    if expand:
        img_done_count = img_done_count + 1
        if img_done_count % 100 == 0:
            print('\r--%d/%d %s'%(img_done_count,len(mark_data), 'images done'), end = '')
        x1, y1, x2, y2 = gt_box
        gt_box_w = x2 - x1 + 1
        gt_box_h = y2 - y1 + 1
        if max(gt_box_w, gt_box_h) < 40 or x1< 0 or y1 < 0:
            #box size比40还小，或者框小于0了，及不必扩展增加数量了
            continue
        for i in range(10):
            #中心移动
            new_box_size = npr.randint(int(min(gt_box_w, gt_box_h) * 0.8),np.ceil(1.25 * max(gt_box_w, gt_box_h)))
            delta_x = npr.randint(-gt_box_w * 0.2, gt_box_w * 0.2)
            delta_y = npr.randint(-gt_box_h * 0.2, gt_box_h * 0.2)
            gt_box_center_x = x1 + gt_box_w/2
            gt_box_center_y = y1 + gt_box_h/2
            new_x1 = int(max(gt_box_center_x + delta_x - new_box_size/2, 0))
            new_y1 = int(max(gt_box_center_y + delta_y - new_box_size/2, 0))
            new_x2 = new_x1 + new_box_size
            new_y2 = new_y1 + new_box_size
            if new_x2 > img_w or new_y2 > img_h:
                continue
               
            new_box = np.array([new_x1, new_y1, new_x2, new_y2])
            new_img = img[new_y1 : new_y2+1, new_x1 : new_x2+1,:]
            new_img_resized = cv2.resize(new_img, (pic_size, pic_size))
            iou = IoU(new_box, np.expand_dims(gt_box, 0))
            if iou > 0.65:
                a_img_a_box_pic_resized.append(new_img_resized)
                marks_rate = np.zeros((5, 2))                
                for index, one in enumerate(marks):
                    rv = ((one[0] - new_x1) / new_box_size,(one[1] - new_y1) / new_box_size)                      
                    marks_rate[index] = rv                      
                a_img_marks_rate.append(marks_rate.reshape(10))
               
                
                
                marks_rate = np.zeros((5, 2))
                landmark_ = a_img_marks_rate[-1].reshape(-1, 2)
                new_rand_box = BBox([new_x1, new_y1, new_x2, new_y2])
                #随机指定下面的进行
                """是否增加一组反转的"""
                if random.choice([0, 1]) > 0:
                    face_flipped, landmark_flipped = flip(new_img_resized, landmark_)
                    face_flipped = cv2.resize(face_flipped,(pic_size, pic_size))
                    a_img_a_box_pic_resized.append(face_flipped)
                    a_img_marks_rate.append(landmark_flipped.reshape(10))
                """是否增加一组镜逆时针旋转5度的，和一组旋转后镜像的"""
                if random.choice([0, 1]) > 0:
                    face_rotated_by_alpha, landmark_rotated = rotate(img, new_rand_box,new_rand_box.rate_to_value_marks(landmark_),5)
                    #逆时针旋转5度                        
                    landmark_rotated = new_rand_box.value_to_rate_marks(landmark_rotated)
                    face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha,(pic_size, pic_size))
                    a_img_a_box_pic_resized.append(face_rotated_by_alpha)
                    a_img_marks_rate.append(landmark_rotated.reshape(10))
                    #镜像
                    face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                    face_flipped = cv2.resize(face_flipped, (pic_size, pic_size))
                    a_img_a_box_pic_resized.append(face_flipped)
                    a_img_marks_rate.append(landmark_flipped.reshape(10))
                """是否增加一组镜顺时针旋转5度的，和一组旋转后镜像的"""
                if random.choice([0, 1]) >= 0:
                    face_rotated_by_alpha, landmark_rotated = rotate(img, new_rand_box, new_rand_box.rate_to_value_marks(landmark_),-5)
                    #逆时针旋转5度                        
                    landmark_rotated = new_rand_box.value_to_rate_marks(landmark_rotated)
                    face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha,(pic_size, pic_size))
                    a_img_a_box_pic_resized.append(face_rotated_by_alpha)
                    a_img_marks_rate.append(landmark_rotated.reshape(10))
                    #镜像
                    face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                    face_flipped = cv2.resize(face_flipped, (pic_size, pic_size))
                    a_img_a_box_pic_resized.append(face_flipped)
                    a_img_marks_rate.append(landmark_flipped.reshape(10))
        
           
        a_img_a_box_pic_resized, a_img_marks_rate = np.asarray(a_img_a_box_pic_resized), np.asarray(a_img_marks_rate)
        
        
        for i in range(len(a_img_a_box_pic_resized)):
            if np.sum(np.where(a_img_marks_rate[i] <= 0, 1, 0)) > 0:
                continue
            if np.sum(np.where(a_img_marks_rate[i] >= 1, 1, 0)) > 0:
                continue
            
            cv2.imwrite(join(path.root,"mark_%d/%d.jpg" % (pic_size,mark_pic_count)), a_img_a_box_pic_resized[i])
            landmarks = map(str, list(a_img_marks_rate[i]))
            f.write(join(path.root, "mark_%d/%d.jpg" % (pic_size,mark_pic_count)) + " -2 "+" ".join(landmarks)+"\n")
            mark_pic_count += 1
            
f.close()
        
 
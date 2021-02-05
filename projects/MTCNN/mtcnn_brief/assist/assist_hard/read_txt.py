#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 08:03:41 2019

@author: q
"""
def read_wider_face_train_bbx_gt(data_path,txt_path):
    allimages_path = []#所有图片的路径
    allimages_boxes = []#所有图片的所有bbox
    with open(txt_path, 'r') as f: 
        kk = 0
        while True:
            #kk += 1#去0个数用于快速检验程序
            """取出路径并收集"""
            image_path = f.readline().strip('\n')
            if kk == 3 or not image_path: break
            image_path = data_path +'/' + image_path
            allimages_path.append(image_path)
            """获取box个数"""
            nums = f.readline().strip('\n')
            """收集box信息"""
            image_boxes = []
            for i in range(int(nums)):
                #左上角坐标，还有宽长
                image_box_info = f.readline().strip('\n').split(' ')
                image_box = [float(image_box_info[i]) for i in range(4)]
                xmin = image_box[0]
                ymin = image_box[1]
                xmax = xmin + image_box[2]
                ymax = ymin + image_box[3]
                image_boxes.append([xmin, ymin, xmax, ymax])    
            allimages_boxes.append(image_boxes)
        print('the total data num is',len(allimages_path))
        data = dict()
        data['allimages_path'] = allimages_path
        data['allimages_boxes'] =  allimages_boxes
        return data
    
    

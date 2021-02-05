#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:00:35 2019

@author: q
标记点txt内容的获取以及对标记点图片采用的镜像选装等增加数量的操作
"""
import os
import numpy as np
import cv2
class BBox(object):
    def __init__(self, bbox):
        self.left = bbox[0]#x1
        self.top = bbox[1]#y1
        self.right = bbox[2]#x2
        self.bottom = bbox[3]#y2
        
        self.x = bbox[0]
        self.y = bbox[1]
        self.w = bbox[2] - bbox[0]
        self.h = bbox[3] - bbox[1]
    def rate_to_value(self, point):
        x = self.x + self.w*point[0]
        y = self.y + self.h*point[1]
        return np.asarray([x, y])
    def rate_to_value_marks(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.rate_to_value(landmark[i])
        return p
    def value_to_rate(self, point):
        """相对于label框的偏置的比例"""
        x = (point[0] - self.x) / self.w
        y = (point[1]- self.y) / self.h
        return np.asarray([x, y])
    def value_to_rate_marks(self, landmark):
        """给出标记位置的偏移量，获得偏移率"""
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.value_to_rate(landmark[i])
        return p
def rate_to_value(box, point):
    x = box[0] + (box[2]-box[0])*point[0]
    y = box[1] + (box[3]-box[1])*point[1]
    return np.asarray([x, y])
def rate_to_value_marks(box, landmark):
    p = np.zeros((len(landmark), 2))
    for i in range(len(landmark)):
        p[i] = rate_to_value(box,landmark[i])
    return p
def value_to_rate(box, point):
    """相对于label框的偏置的比例"""
    x = (point[0] - box[0]) / (box[2]-box[0])
    y = (point[1] - box[1]) / (box[3]-box[1])
    return np.asarray([x, y])
def value_to_rate_marks(box, landmark):
    """给出标记位置的偏移量，获得偏移率"""
    p = np.zeros((len(landmark), 2))
    for i in range(len(landmark)):
        p[i] = value_to_rate(box,landmark[i])
    return p
"""
 data_path根目录，txt根目录图片的信息
返回的是一个list，每一项都是图片路径，方框和mark
"""
def getDataFromTxt(txt, data_path):
    with open(txt, 'r') as  fd:
        lines = fd.readlines()
    result = []
    for line in lines:
        line = line.strip()
        components = line.split(' ')
        img_path = os.path.join(data_path, components[0]).replace('\\','/')
        bbox = (components[1],components[3], components[2],components[4])
        #lfw5590他的box是 x1,x2,y1,y2所以要改一下
        bbox = [float(_) for _ in bbox]
        bbox = list(map(int, bbox))
        landmark = np.zeros((5, 2))
        for index in range(0, 5):
            rv = (float(components[5+2*index]),
                  float(components[5+2*index+1]))
            landmark[index] = rv
        result.append((img_path,BBox(bbox),landmark))
    return result
   
def flip(face, landmark):
    """
    图像镜像，并且mark坐标镜像
    """
    face_flipped_by_x = cv2.flip(face, 1)
    #mirror
    landmark_ = np.asarray([(1-x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
    landmark_[[3, 4]] = landmark_[[4, 3]]#left mouth<->right mouth
    return (face_flipped_by_x, landmark_)
def rotate(img, bbox, landmark, alpha):
    """
        given a face with bbox and landmark, rotate with alpha
        and return rotated face with bbox, landmark (absolute position)
    """
    center = ((bbox.left+bbox.right)/2, (bbox.top+bbox.bottom)/2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)#旋转的矩阵
    
    #whole image rotate
    #pay attention: 3rd param(col*row)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat,(img.shape[1],img.shape[0]))
    landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark])
    #crop face 
    face = img_rotated_by_alpha[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
    return (face, landmark_)

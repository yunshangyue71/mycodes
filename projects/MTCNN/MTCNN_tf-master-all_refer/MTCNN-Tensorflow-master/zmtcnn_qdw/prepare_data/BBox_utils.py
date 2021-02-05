# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 11:34:43 2019

@author: QDW
"""

import os 
from os.path import join, exists
import time
import cv2
import numpy as np

class BBox(object):
    def __init__(self, bbox):
        self.left = bbox[0]#x1
        self.top = bbox[1]#y1
        self.right = bbox[2]#x2
        self.bottom = bbox[3]#y2
        #这里他把上下标记错了
        
        self.x = bbox[0]
        self.y = bbox[1]
        self.w = bbox[2] - bbox[0]
        self.h = bbox[3] - bbox[1]
    def expand(self, scale=0.05):
        """长宽个变大10%"""
        bbox = [self.left, self.right, self.top, self.bottom]
        #改变形式，[x1,x2,y1,y2]
        bbox[0] -= int(self.w * scale)
        bbox[1] += int(self.w * scale)
        bbox[2] -= int(self.h * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)
    def project(self, point):
        """相对于label框的偏置的比例"""
        x = (point[0] - self.x) / self.w
        y = (point[1]- self.y) / self.h
        return np.asarray([x, y])
    def reproject(self, point):
        """给出偏置比例，获得位置"""
        x = self.x + self.w * point[0]
        y = self.y + self.h * point[1]
        return np.asarray([x, y])
    def projectLandmark(self, landmark):
        """给出标记位置的偏移量，获得偏移率"""
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmatk)):
            p[i] = self.project(landmark[i])
        return p
    def reprojectLandmark(self, landmark):
        """给出标记位置的偏移率，获得偏移量"""
        p = np.zeros((len(lanmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p
    def subBBox(self, leftR, rightR, topR, bottomR):
        leftDelta = self.w  * leftR
        rightDelta = self.w * rightR
        topDelata = self.h * topR
        bottomDelta = self.h * bootomR
        
        left = self.left + leftDelta
        right = self.left + rightDelta
        top = self.top + topDelta
        bottom = self.top + bootomDelta
        return BBox([left, right, top, bottom])
    
        
def getDataFromTxt(txt, data_path, with_landmark = True):
"""
txt:获取label框和标记点位置的文件
data_path:就是包含积极框文件夹，消极框文件夹，部分框文件夹的文件夹
with_landmark:是否需要输出标记点的位置信息呢

不需要输出标记点位置信息：
result[img_path, BBox(bbox)]
result.append((img_path,BBox(bbox),landmark))
img_path:label框和标记点位置信息都有的数据集位置
"""
    with open(txt, 'r') as  fd:
    #读取txt文件中的所有的行。
        lines = fd.readlines()
        
    result = []
    for line in lines:
        #遍历所有行
        line = line.strip()
        #去掉首尾的空白字符，包括换行符
        components = line.split(' ')
        #将每一行用空白作为列表分割
        img_path = os.path.join(data_path, 
                                components[0].replace('\\','/'))
        #将双斜杠换未单反斜杠
        #获取label框和标记点都有的图像的图像的路径
        bbox = (components[1],components[3],
                components[2],components[4])
        #bbox左下右上，txt，左右下上
        bbox = [float(_) for _ in bbox]
        bbox = list(map(int, bbox))
        #不明白但是不妨碍
        if not with_landmark:
        #如果不用landmark就直接输出
            result.append((img_path, BBox(bbox)))
            continue
        landmark = np.zeros((5, 2))
        #标记点的位置列表（空）
        for index in range(0, 5):
        #将txt文件中的5个标记点的坐标输出矩阵的形式
            rv = (float(components[5+2*index]),
                  float(components[5+2*index+1]))
            landmark[index] = rv
        result.append((img_path,BBox(bbox),landmark))
        #文件的1324对应的是左下右上
    return result
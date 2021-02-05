# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:59:15 2019

@author: QDW
"""

import os 
import random 
from os.path import join, exists
import cv2
import numpy as np
import numpy.random as npr
from utils import IoU
from Landmark_utils import rotate, flip
from BBox_utils import getDataFromTxt, BBox

def GenerateData(ftxt, data_path, net, argument = False):
"""

"""
    """根据网络名来判断size大小"""
    if net == 'PNet':
        size = 12
    elif net == 'RNet':
        size = 24
    elif net == 'ONet':
        size = 48
    else:
        print('不是PNet，RNet，ONet，net类型错误，请检查')
        return
    
    image_id = 0
    f = open(os.path.join(OUTPUT, 'landmark_%s_aug.txt'%(size)), 'w')
    data = getDataFromTxt(ftxt, data_path = data_path)
    #将txt的1234对应BBOx的0123，左上右下
    #列表 带有标记位置的路径，label框，标记点坐标
    idx = 0#计数图片标记点，label狂的信息提取出了多少
    for (imgPath, bbox, landmarkGt) in data:
    """遍历txt文件中的所有数据"""
        F_imgs = []#每张照片的所有人脸图像
        F_landmarks = []#每张照片的所有标记点
        img = cv2.imread(imgPath)
        #读取图像路径的图片
        assert(img is not None)
        img_h, img_w, img_c = img.shape
        gt_box = np.array([bbox.left, bbox.top,
                           bbox.right, bbox.bottom])
        #gt_box:左下右上，0123 txt，左右下上
        """作者把上下名称颠倒了"""
        f_face = img[bbox.top : bbox.bootom+1, bbox.left : bbox.right+1]
        #像素点
        f_face = cv2.resize(f_face, (size, size))
        #读取人脸图像
        landmark = np.zeros((5, 2))
        for index, one in enumerate(landmarkGt):
        """遍历每一个图像的5个标记点，
        index：0-4
        one: 标记点的位置
        """
            rv = ((one[0] - gt_box[0]) / (gt_box[2] - gt_box[0]), 
                  (one[1] - gt_box[1]) / (gt_box[3] - gt_box[1]))
            #关键点的位置信息转换成比例，距离左下角的距离和边长的比例
            landmark[index] = rv
        F_imgs.append(f_face)
        F_landmarks.append(lanmark.reshap(10))
        landmark = np.zeros((5, 2))
        if argument:
            idx = idx + 1
            if idx % 100 == 0:
                print(idx, 'images done')
            x1, y1, x2, y2 = gt_box
            #label框的坐上
            gt_w = x2 - x1 + 1
            gt_h = y2 - y1 + 1
            if max(gt_w, gt_h) < 40 or x1< 0 or y1 < 0:
            #如果出错就舍弃这张图片
                continue
            for i in range(10):
               bbox_size = npr.randint(int(min(gt_w, gt_h) * 0.8),
                                       np.ceil(1.25 * max(gt_w, gt_h)))
               #偏移量是少20%的最小值，多20%的最大值
               delta_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
               delta_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
               nx1 = int(max(x1 + gt_w/2 - bbox_size/2 + delta_x, 0))
               ny1 = int(max(y1 + gt_h/2 - bbox_size/2 + delta_y, 0))
               #label框随机移动后中心点的位置在，构造label框和移动后
               #label框中心重合，但是x1， y1大于0
               nx2 = nx1 + bbox_size
               ny2 = ny1 + bbox_size
               if nx2 > img_w or ny2 > img_h:
               #x2,y2还必须在图片内
                   continue
               
               crop_box = np.array([nx1, ny1, nx2, ny2])
               cropped_im = img[ny1 : ny2+1, nx1 : nx2+1,:]
               resized_im = cv2.resize(cropped_im, (size, size))
               iou = IoU(crop_box, np.expand_dims(gt_box, 0))
               if iou > 0.65:
                   F_imgs.append(resized_im)
                   #这张图片前期添加了一个label框的像素图像了
                   #现在添加进去的是构造label框的图像
                   for index, one in enumerate(landmarkGt):
                       #将关键点相对于构造label框的信息添加进去
                       rv = ((one[0] - nx1) / bbox_size,
                             (one[1] - ny1) / bbox_size)
                       landmark[index] = rv
                      
                    F_landmarks.append(landmark.reshape(10))
                    #这张图片前期添加了一个label框的关键点的信息了
                   #现在添加进去的是构造label框的关键点的信息了
                    landmark = np.zeros((5, 2))
                    landmark_ = Flandmarks[-1].reshape(-1, 2)
                    bbox = BBox([nx1, ny1, nx2, ny2])
                    """随机镜像"""
                    if random.choice([0, 1]) > 0:
                        face_flipped, landmark_flipped = \
                        flip(resized_im, landmark)
                        face_flipped = cv2.resize(face_flipped,
                                                  (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
                    """随机反转"""
                    if random.choice([0, 1]) > 0:
                        face_roteted_by_alpha, landmark_rotated = \
                        rotate(img, bbox, 
                               bbox.reprojectLandmark(landmark_),5)
            
            
    
    
    
    f.close()
        
    
    pass

if __name__ == '__main__':
#这里应该设置成有标记点的数据集的位置
    
    with open('DATA_routing.txt', 'r') as f:
        img_root = f.readline()
        img_root = img_root.strip()
        #print('img_root:',img_root)
        mtcnn_results = f.readline()
        mtcnn_results = mtcnn_results.strip()
        prepare_data_path = os.path.join(
                mtcnn_results,'mtcnn_prepared_data')
        data_path = prepare_data_path
        #print('mtcnn_results:',mtcnn_results)
        dsrdir = os.path.join(prepare_data_path, 
                              '12/train_PNet_andmark_aug')
        OUTPUT = os.path.join(prepare_data_path, 
                              '12')
    
    if not exists(OUTPUT):
        os.mkdir(OUTPUT)
    if not exists(dstdir):
        os.mkdir(dstdir)
    assert (exists(dstdir) and exists(OUTPUT))
    net = "PNet"
    train_txt = 'trainImageList.txt'
    imgs, landmarks = GenerateData(train_txt, 
                                   data_path, net, argument=True)
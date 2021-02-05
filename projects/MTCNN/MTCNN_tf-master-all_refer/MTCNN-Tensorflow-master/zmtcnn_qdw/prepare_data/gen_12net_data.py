# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:21:49 2019

@author: QDW
"""
import os 
import cv2
import numpy as np
import numpy.random as npr

from prepare_data.utils import IoU
"""一路径设置"""
anno_file = 'wider_face_train.txt'
im_dir = '../../DATA/WIDER_train/images'
pos_save_dir = '../../DATA/12/positive'
part_save_dir = '../../DATA/12/part'
neg_save_dir = '../../DATA/12/negative'
save_dir = '../../DATA/12'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(pos_save_dir):
    os.mkdir(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.mkdir(neg_save_dir)
    
f1 = open(os.path.join(save_dir, 'pos_12.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_12.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_12.txt'), 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print('%d pictures in total' %num)
p_idx = 0
n_idx = 0#训练图像中所有消极框的个数
d_idx = 0
idx = 0
box_idx = 0
for annotation in annotations:
    """从wider_face_train.txt获取路径以及方框"""
    annotation = annotation.strip().split(' ')
    #srtip将annotation这个字符串的前面后面的空格去掉
    #split根据空格将字符串分割成一个列表
    im_path = annotation[0]
    #这个是类似于地址的地方
    bbox = list(map(float, annotation[1:]))
    #对每个列表元素设置位浮点数，新的bbox（4个数字）
    boxes = np.array(bbox, dtype = np.float32).reshape(-1, 4)
    #转换成numpy数组,并将编程2维度的了
    #有可能有两个label框
    img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
    #读取wider_face路径下的训练数据
    
    """寻找50个消极框"""
    idx += 1
    #消极框计数
    height, width, channel = img.shape
    neg_num = 0
    #每个图片的消极框计数
    while neg_num < 50:
        size = npr.randint(12, min(width, height) / 2)
        #随机选择一个size（12-）
        nx = npr.randint(0,width - size)
        ny = npr.randint(0, height - size)
        #随机选择一个坐标，要使得方框在图片内
        crop_box = np.array([nx, ny, nx + size, ny + size])
        #将方框左下角以及方框尺寸形成一个list
        Iou = IoU(crop_box, boxes)
        #计算方框和label方框的重叠度
        cropped_im = img[ny : ny+size, nx : nx+size, :]
        #获取选中方框的内容
        resized_im = cv2.resize(cropped_im, (12, 12),
                                interpolation = cv2.INTER_LINEAR)
        #将方框的内容变为12*12大小
        if np.max(Iou) < 0.3:
            #随机框和label框的重合度<0.3，就判定为消极框
            save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            #这个是所有图片的消极框的个数
            f2.write('../../DATA/12/negative/%s.jpg'%n_idx +' 0\n')
            #在neg_12.txt文件中写入生成消极框的路径
            cv2.imwrite(save_file, resized_im)
            #在上面路径上写入12*12的图片的内容
            n_idx += 1
            #全部的消极框计数+1
            neg_num += 1
            #本图片的消极框计数+1
    
    for box in boxes:
        x1, y1, x2, y2 = box
        #label框的左下和右上的坐标取出
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        #计算宽和高
        if max(w, h) < 20 or x1 < 0 or y1 <0:
        #如果左下的坐标有负值或者label框的宽和高均小于20，
        #就舍弃这个label框
            continue
        for i in range(5):
        #循环5次
            size = npr.randint(12, min(width, height) / 2)
            #随机选择也给尺寸（12-）
            delta_x = npr.randint(max(-size, -x1), w)
            #x的位移量最大时方框的宽（x1最大可以向右位移w）
            #x的最小位移量，也就是向左的最大位移量是max（-size，-x1）
            #既要label框有重合，又不能移动到负值区域去。
            delta_y = npr.randint(max(-size, -y1), h)
            #
            nx1 = int(max(0, x1 + delta_x))
            ny1 = int(max(0, y1 + delta_y))
            #如果位移到了负值，就选择远点
            if nx1 + size > width or ny1 + size > height:
            #如果新的方框在img之外了那么新的方框失败
                continue
            crop_box = np.array([nx1, ny1, nx1+size, ny1+size])
            #构造的label框和label框的重叠度
            Iou = IoU(crop_box, boxes)
            cropped_im = img[nyq:ny1 + size, nx1: nx1+size, :]
            resized_im = cv2.resize(cropped_im, (12, 12),
                                    interpolation = cv2.INTER_LINNEAR)
            if np.max(Iou) < 0.3:
                save_file = os.path.join(neg_save_dir,
                                         "%s.jpg"%n_idx)
                f2.write("../../DATA/12/negative/%s.jpg"%n_idx+' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
        
        for i in range(20):
            size = npr.randint(int(min(w, h) * 0.8), 
                               np.ceil(1.25 * max(w, h)))
            #构造的label框尺寸是label框
            if  w<5:
            #如果宽度小于5就跳出这次随机
                print(w)
                continue
            delta_x = npr.randint(-w*0.2, w*0.2)
            delta_y = npr.randint(-h*0.2, h*0.2)
            nx1 = int(max(x1 + w/2 + delta_x - size/2, 0))
            ny1 = int(max(y1 + h/2 + delta_y - size/2, 0))
            nx2 = nx1 + size
            ny2 = ny1 + size
            
            if nx2 > width or nyw > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])
            offset_x1 = (x1-nx1) / float(size)
            offset_y1 = (y1-ny1) / float(size)
            offset_x2 = (x2-nx2) / float(size)
            offset_y2 = (y2-ny2) / float(size)
            
            cropped_im = img[ny1:ny2, nx1:nx2, :]
            resized_im = cv2.resize(cropped_im, (12, 12),
                                    interpolation = cv2.INTER_LINEAR)
            box_ = box.reshape(1, -1)
            iou = IoU(crop_box,box_)
            if iou >= 0.65:
                save_file = os.path.join(pos_save_dir,
                                         "%s.jpg"%p_idx)
                f1.write("../''/DATA/12/positive/%s.jpg"%p_idx +
                         '-1 %.2f %.2f %.2f %.2f\n'
                         %(offset_x1, offset_y1, offset_x2,offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif iou >= 0.4；:
                save_file = os.path.join(part_save_dir,
                                         "%s.jpg"%d_idx)
                f3.write("../../DATA/12/%s.jpg"%d_idx +
                         '-1 %.2f %.2f %.2f %.2f\n'
                         %(offset_x1, offset_y1, offset_x2,offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
        box_idx += 1
        if idx % 100 == 0:
            print("%s images done,pos:%s part: %s neg: %s"
                  %(idx, p_idx, d_idx, n_idx))
f1.close()
f2.close()
f3.close()
        

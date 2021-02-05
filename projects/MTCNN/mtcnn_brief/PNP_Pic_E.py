#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function:prepare pos, neg, part picture for PNet,RNet,ONet,which not use hard sampling,
         so i use the symbol E which means easy.
"""
import os 
import cv2
import numpy as np
import numpy.random as npr

from path_and_config import path
from assist.iou import IoU

pic_size = 12#PNet

#pic_size大小的pos,neg,part图片保存的位置    
pos_save_dir = os.path.join(path.root,'pic_%d_E/pic_%d_pos_E'%(pic_size,pic_size))
neg_save_dir = os.path.join(path.root,'pic_%d_E/pic_%d_neg_E'%(pic_size,pic_size))
part_save_dir = os.path.join(path.root,'pic_%d_E/pic_%d_part_E'%(pic_size,pic_size))
if not os.path.exists(pos_save_dir):
    os.makedirs(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.mkdir(neg_save_dir)

#pos,neg,part pic_size 大小的图片对应的txt文件
txt_dir = os.path.join(path.root,'pic_txt')
if not os.path.exists(txt_dir):
    os.mkdir(txt_dir)
pos_txt_dir = os.path.join(path.root,'pic_txt/pic_%d_pos_E.txt'%(pic_size))
neg_txt_dir = os.path.join(path.root,'pic_txt/pic_%d_neg_E.txt'%(pic_size))
part_txt_dir = os.path.join(path.root,'pic_txt/pic_%d_part_E.txt'%(pic_size))
#widerface原始的标记文件
pnp_txt = 'txt/wider_face_train.txt'
with open(pnp_txt, 'r') as f:
    infos = f.readlines()
    print('The number of pictures which used for making PNP pics is ', len(infos))
f1 = open(pos_txt_dir, 'w')
f2 = open(neg_txt_dir, 'w')
f3 = open(part_txt_dir, 'w')
pos_count = 0; neg_count = 0; part_count = 0
infos_done_count = 0; all_images_all_boxes_count= 0
"""执行核心程序"""
for info in infos:
    info = info.strip().split(' ')
    im_path = info[0]#路径

    box = list(map(float, info[1:]))#框的信息
    boxes = np.array(box, dtype = np.float32).reshape(-1, 4)
    
    img = cv2.imread(os.path.join(path.pnp, im_path + '.jpg'))  
    height, width, channel = img.shape
    
    """每张图片找50个pic_neg_pic_size"""
    neg_while = 0
    while neg_while < 50:
        new_size = npr.randint(pic_size, min(width, height) / 2)#??
        new_x1 = npr.randint(0,width -new_size)
        new_y1 = npr.randint(0, height - new_size)
        new_box = np.array([new_x1, new_y1, new_x1+new_size, new_y1+new_size])
        Iou = IoU(new_box, boxes)#计算new_box和image中各个boxes的重叠度，返回一个list
        new_image = img[new_y1 : new_y1+new_size, 
                        new_x1 : new_x1+new_size,
                        :]
        new_image_resized = cv2.resize(new_image, (pic_size, pic_size),
                                interpolation = cv2.INTER_LINEAR)
        if np.max(Iou) < 0.3:#如果构造的框和这个图片中的每个box重叠度都小于0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg"%neg_count)
            f2.write(neg_save_dir + '/%s.jpg'%neg_count +' 0\n')
            cv2.imwrite(neg_save_dir + '/%s.jpg'%neg_count, new_image_resized)
            neg_count += 1
            neg_while += 1
    
    for box in boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        if max(w, h) < 20 or x1 < 0 or y1 <0:
            continue
        for i in range(5):
            size = npr.randint(pic_size, min(width, height) / 2)
            delta_x = npr.randint(-size, w)
            delta_y = npr.randint(-size, h)
            new_x1 = int(x1 + delta_x)
            new_y1 = int(y1 + delta_y)
            if new_x1 < 0: new_x1 = 0
            if new_y1 < 0: new_y1 = 0
            if new_x1 + size > width or new_y1 + size > height:
                continue
            new_box = np.array([new_x1, new_y1, new_x1+size, new_y1+size])
            Iou = IoU(new_box, boxes)
            new_image = img[new_y1:new_y1 + size, new_x1: new_x1+size, :]
            new_image_resized = cv2.resize(new_image, (pic_size, pic_size),
                                    interpolation = cv2.INTER_LINEAR)
            if np.max(Iou) < 0.3:
                save_file = os.path.join(neg_save_dir,"%s.jpg"%neg_count)
                f2.write(neg_save_dir + "/%s.jpg"%neg_count+' 0\n')
                cv2.imwrite(neg_save_dir + "/%s.jpg"%neg_count, new_image_resized)
                neg_count += 1
       
        for i in range(20):
            size = npr.randint(int(min(w, h) * 0.8),  np.ceil(1.25 * max(w, h)))
            if  w<5:
                print(w)
                continue
            center_x = x1 + w/2
            center_y = y1 + h/2
            delta_center_x = npr.randint(-w*0.2, w*0.2)
            delta_center_y = npr.randint(-h*0.2, h*0.2)
            new_center_x = center_x + delta_center_x
            new_center_y = center_y + delta_center_x 
            new_x1 = int(max(new_center_x - size/2, 0))
            new_y1 = int(max(new_center_y - size/2, 0))
            new_x2 = new_x1 + size
            new_y2 = new_y1 + size
            
            if new_x2 > width or new_y2 > height:
                continue
            new_box = np.array([new_x1, new_y1, new_x2, new_y2])
            offset_x1_rate = (x1-new_x1) / float(size)
            offset_y1_rate = (y1-new_y1) / float(size)
            offset_x2_rate = (x2-new_x2) / float(size)
            offset_y2_rate = (y2-new_y2) / float(size)
            
            new_image = img[new_y1:new_y2, new_x1:new_x2, :]
            new_image_resized = cv2.resize(new_image, (pic_size,pic_size),
                                    interpolation = cv2.INTER_LINEAR)
            #注意这里的cv2.resize的img[y在前面]
            box_ = box.reshape(1, -1)
            iou = IoU(new_box,box_)
            if iou >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg"%pos_count)
                f1.write(pos_save_dir + "/%s.jpg"%pos_count +' 1 %.2f %.2f %.2f %.2f\n' %(offset_x1_rate, offset_y1_rate, offset_x2_rate,offset_y2_rate))
                cv2.imwrite(pos_save_dir + "/%s.jpg"%pos_count, new_image_resized)
                pos_count += 1
            elif iou >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg"%part_count)
                f3.write(part_save_dir + "/%s.jpg"%part_count + ' -1 %.2f %.2f %.2f %.2f\n'
                         %(offset_x1_rate, offset_y1_rate, offset_x2_rate,offset_y2_rate))
                cv2.imwrite(part_save_dir + "/%s.jpg"%part_count, new_image_resized)
                part_count += 1
        all_images_all_boxes_count += 1
    if infos_done_count % 100 == 0:
        print("%s/%s images done,pos:%s part: %s neg: %s"
                  %(infos_done_count, len(infos),pos_count,part_count,neg_count))
    infos_done_count += 1
            
f1.close()
f2.close()
f3.close()

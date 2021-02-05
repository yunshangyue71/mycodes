#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle as pickle
import cv2
import numpy as np

from path_and_config import path
from assist.assist_hard.convert_to_square import convert_to_square
from assist.iou import IoU
from assist.assist_hard.read_txt import read_wider_face_train_bbx_gt
from assist.assist_hard.iter_data import TestLoader
from assist.assist_hard.singledetector import SingleDetector
from assist.assist_hard.batchdetector import BatchDetector
from assist.assist_train.forward import P_Net,R_Net,O_Net
from assist.assist_hard.mtcnndetect import MtcnnDetector

def t_net(models_pathes, epoch, batch_size, thresh, min_face_size, stride ,scale_factor,
          pic_size,pnp_pic_hard_sample,
          slide_window = False,shuffle = False,vis  = False): 
    """数据准备""" 
    pnp_txt = 'txt/wider_face_train_bbx_gt.txt'
    data = read_wider_face_train_bbx_gt(path.pnp, pnp_txt)
    iter_im = TestLoader(data['allimages_path'])#迭代器地址
    """model路径，检测器""" 
    models_path=['%s-%s'%(x,y) for x,y in zip(models_pathes, epoch)]
    detectors=[None,None,None]
    if pic_size == 24 or pic_size == 48 or pic_size == 0 : 
        PNet = SingleDetector(P_Net, models_path[0])
        detectors[0] = PNet
    if pic_size == 48 or pic_size == 0: 
        RNet = BatchDetector(R_Net, 24, batch_size[1],models_path[1])
        detectors[1] = RNet
    if pic_size == 0 : #这个就相当于测试了，在hardsample中没有使用到
        ONet = BatchDetector(O_Net, 48, batch_size[2],models_path[2])
        detectors[2] = ONet
    
    mtcnn_detector = MtcnnDetector(detectors,
                                   min_face_size,#20
                                   stride,#2
                                   thresh,#[0.6,0.7,0.7]
                                   scale_factor)#0.79窗口缩小比例
    detections, _ = mtcnn_detector.detect_face(iter_im)
    """第三步，图片保存""" 
    save_path = os.path.join(path.root, 'pic_%d%s'%(pic_size,pnp_pic_hard_sample))
    if not os.path.exists(save_path):     os.mkdir(save_path)
    save_file = os.path.join(save_path,'detections.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(detections, f, 1)#将detection放到detection.pkl文件中去
    print("%s测试完成开始OHEM"% pic_size)
    save_hard_example(pic_size,pnp_pic_hard_sample, data, save_path)
def save_hard_example(pic_size, pnp_pic_hard_sample, data, save_path):
    ims_path_list = data['allimages_path']
    ims_boxes_list = data['allimages_boxes']
    num_of_images = len(ims_path_list)
    print('##############开始写进txt，和图片文件夹#########################')
    print('processing %d images in total'% num_of_images)
    
    neg_txt = os.path.join(path.root,'pic_txt/pic_%d_neg%s.txt' %(pic_size,pnp_pic_hard_sample))
    pos_txt = os.path.join(path.root,'pic_txt/pic_%d_pos%s.txt' %(pic_size,pnp_pic_hard_sample))
    part_txt =os.path.join(path.root,'pic_txt/pic_%d_part%s.txt'%(pic_size,pnp_pic_hard_sample))                                                       
    fneg = open(neg_txt, 'w');fpos = open(pos_txt, 'w');fpart = open(part_txt, 'w')
    
    det_boxes = pickle.load(open(os.path.join(save_path, 'detections.pkl'),'rb'))
    print(len(det_boxes))
    print(num_of_images)
    assert len(det_boxes) == num_of_images, 'incorrect detections or ground truths'
    ims_n_idx = 0
    ims_p_idx = 0
    ims_d_idx = 0
    image_done = 0
        #每张照片的              #data地址，构造的框，    data的框
    for im_path, im_dets,im_boxes in zip(ims_path_list,det_boxes,ims_boxes_list):
        im_boxes = np.array(im_boxes, dtype = np.float32).reshape(-1, 4)
        if im_dets.shape[0] == 0:#resized一共
            continue
        img = cv2.imread(im_path)
        im_dets = convert_to_square(im_dets)
        im_dets[:,0:4] = np.round(im_dets[:, 0:4])
        im_neg_num = 0
        for box in im_dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left +1
            height = y_bottom - y_top + 1
            if width < 20 or x_left<0 or y_top < 0 or x_right>img.shape[1] - 1 \
            or y_bottom > img.shape[0] - 1:
                continue
            Iou = IoU(box, im_boxes)#构造的框和label框进行重叠度比较
            cropped_im = img[y_top:y_bottom+1,x_left:x_right+1,:]
            resized_im = cv2.resize(cropped_im,(pic_size, pic_size),
                                    interpolation = cv2.INTER_LINEAR)
            if np.max(Iou)<0.3 and im_neg_num<60:
                save_file = os.path.join(neg_dir, '%s.jpg'%ims_n_idx)
                fneg.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                ims_n_idx += 1
                im_neg_num += 1
            else:
                idx = np.argmax(Iou)
                assigned_gt = im_boxes[idx]
                x1,y1,x2,y2 = assigned_gt
                offset_x1 = (x1-x_left)/float(width)
                offset_y1 = (y1- y_top)/float(height)
                offset_x2 = (x2 - x_right)/float(width)
                offset_y2 = (y2 - y_bottom)/float(height)
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_dir, '%s.jpg'%ims_p_idx)
                    fpos.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' %(offset_x1, offset_y1,offset_x2,offset_y2))
                    cv2.imwrite(save_file,resized_im)
                    ims_p_idx += 1
                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_dir,'%s.jpg'%ims_d_idx)
                    fpart.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' %
                                    (offset_x1, offset_y1,offset_x2,offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    ims_d_idx += 1
                    
        if image_done % 100 == 0:
            print('\r>>%d/%d images written,pos num: %d;neg num: %d;part num:%d' % \
                  (image_done,num_of_images,ims_p_idx,ims_n_idx,ims_d_idx), end = '')
        image_done = image_done+1
                
    fneg.close()
    fpart.close()
    fpos.close()
                    
############################################################################## 
if __name__ == '__main__':
    pic_size = 48#RNet#需要生成图片的尺寸
    #pic_size = 48#ONet
    choice = ['_E','_HE','_HHE']
    #if pic_size == 12: c = 0
    if pic_size == 24: c = 1
    if pic_size == 48: c = 2
    pnp_pic_hard_sample = choice[c] 
    """路径设置"""
    pnp_dir = os.path.join(path.root,'pic_%d%s'%(pic_size, pnp_pic_hard_sample))
    pos_dir = os.path.join(pnp_dir,'pic_%d_pos%s'%(pic_size,pnp_pic_hard_sample))
    neg_dir = os.path.join(pnp_dir,'pic_%d_neg%s'%(pic_size,pnp_pic_hard_sample))
    part_dir = os.path.join(pnp_dir,'pic_%d_part%s'%(pic_size,pnp_pic_hard_sample))
    for dir_path in [neg_dir, pos_dir, part_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    P_model_paths = os.path.join(path.root,'train_model/pic_12_E/pic_12_E')
    R_model_paths = os.path.join(path.root,'train_model/pic_24_HE/pic_24_HE')
    O_model_paths = os.path.join(path.root,'train_model/pic_48_HHE/pic_48_HHE')
    models_pathes=[P_model_paths,R_model_paths,O_model_paths]
    """参数指定"""
    epoch=[18, 14, 16]
    batch_size=[2048, 256, 16]
    thresh=[0.6, 0.7, 0.7]
    min_face_size = 20
    stride=2
    scale_factor = 0.79
    print(' pos dir: ',pos_dir)
    print(' neg dir: ',neg_dir)
    print(' part dir: ',part_dir)
    print(' P model paths: ',P_model_paths)
    print(' R model paths: ',R_model_paths)
    print(' O model paths: ',O_model_paths)
    print('epoch : ',epoch)
    print(' batch size: ', batch_size)
    print(' thresh: ',thresh)
    print('min face size : ',min_face_size)
    print(' stride: ',stride)
    print(' scale_factor: ',scale_factor)
    print('###################################################')
    t_net(models_pathes, epoch, batch_size, thresh, min_face_size, stride,scale_factor,#以默认
          pic_size, pnp_pic_hard_sample,
          slide_window=False, shuffle=True, vis = True)#未使用
    
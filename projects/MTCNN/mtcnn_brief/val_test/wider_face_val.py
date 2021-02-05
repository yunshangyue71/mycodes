#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#RNet
import numpy as np 
import os
import cv2

from path_and_config import path 
from assist.assist_train.forward import P_Net,R_Net,O_Net
from assist.assist_hard.mtcnndetect import MtcnnDetector
from assist.assist_hard.singledetector import SingleDetector
from assist.assist_hard.batchdetector import BatchDetector

def run(wider_val_txt,wider_val_dir,output_file,detectors,mtcnn_detector):
    image_info = read_txt(wider_val_txt)
    current_event = ''
    save_path = ''
    idx = 0
    for item in image_info:
        idx += 1
        image_file_name = os.path.join(wider_val_dir,item[0],item[1])
        if current_event != item[0]:
            current_event = item[0]
            save_path = os.path.join(output_file,item[0])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print('\ncurrent_path:', current_event)
            img = cv2.imread(image_file_name)
            all_boxes, _= detect_single_image(img,detectors,mtcnn_detector)
            
            f_name = item[1].split('.jpg')[0]
            dets_file_name = os.path.join(save_path, f_name +'.txt')
            fid = open(dets_file_name,'w')
            boxes = all_boxes[0]
            if boxes is None:
                fid.write(item[1] + '\n')
                fid.write(str(1)+ '\n')
                fid.write('%f %f %f %f %f\n'%(0,0,0,0,0.99))
                continue
            fid.write(item[1] + '\n')
            fid.write(str(len(boxes)) + '\n')
            for box in boxes:
                fid.write('%f %f %f %f %f\n'%(
                        float(box[0]),float(box[1]),float(box[2]-box[0] +1),
                        float(box[3]-box[1]+1),box[4]))
            fid.close()
        if idx %10 == 0:
            print('\r%d/%d image made'%(idx,len(image_info)),end = '')
            
def detect_single_image(img,detectors,mtcnn_detector):
    all_boxes = []
    marks = []
    #pnet
    
    if detectors[0]:
        boxes, boxes_c, mark = mtcnn_detector.detect_pnet(img)
        if boxes_c is None:
            print('boxes_c is None...')
            all_boxes.append(np.array([]))
            marks.append(np.array([]))
            return all_boxes,marks
        #print('pnet boxes_c:',boxes_c.shape)
        if boxes_c is None:
            print('boxes_c is None agter Pnet')
    #rnet
    if detectors[1] and not boxes_c is None:
        boxes, boxes_c,mark = mtcnn_detector.detect_rnet(img,boxes_c)
        if boxes_c is None:
            print('boxes_c is None...')
            all_boxes.append(np.array([]))
            marks.append(np.array([]))
            return all_boxes,marks
        #print('rnet boxes_c:',boxes_c.shape)
        if boxes_c is None:
            print('boxes_c is None agter Rnet')
    #onet
    if detectors[2] and not boxes_c is None:
        boxes, boxes_c,mark = mtcnn_detector.detect_rnet(img,boxes_c)
        if boxes_c is None:
            print('boxes_c is None...')
            all_boxes.append(np.array([]))
            marks.append(np.array([]))
            return all_boxes,marks
        if boxes_c is None:
            print('boxes_c is None agter Onet')
    all_boxes.append(boxes_c)
    marks.append(mark)
        
    return all_boxes,marks
    
        
        
def read_gt_box(raw_list):
    list_len = len(raw_list)
    box_num = (list_len - 1) // 4
    idx = 1
    boxes = np.zeros((box_num, 4),dtype = int)
    for i in range(4):
        for j in range(box_num):
            boxes[j][i] = int(raw_list[idx])
            idx += 1
    return boxes

def read_txt(txt_path):
    with open(txt_path,'r') as f :
        image_info = []
        for line in f:
            ct_list = line.strip().split(' ')
            path = ct_list[0]
            path_list = path.split('\\')
            event = path_list[0]
            name = path_list[1]
            boxes = read_gt_box(ct_list)
            image_info.append([event,name, boxes])
    print('total number of images in validation set :',len(image_info))
    return image_info

if __name__ == '__main__':
    pic_size = 48

    choice = ['_E','_HE','_HHE']
    if pic_size == 24: c = 0; print('we will test PNet')
    if pic_size == 48: c = 1; print('we will test RNet')
    if pic_size == 0:  c = 2; print('we will test ONet')
    pnp_pic_hard_sample = choice[c]  
    thresh = [0.3,0.1,0.7]
    min_face_size = 20
    stride = 2
    scale_factor = 0.79
    
    #val图像、txt地址，输出图像地址
    wider_val_dir = path.WIDER_VAL
    wider_val_txt = 'wider_face_val.txt' 
    output_file = os.path.join(path.root,'val_test/wider_face_val')
    if not os.path.exists(output_file): os.makedirs(output_file)
    #model path
    prefix = path.root +'/train_model'
    pre_dir = [prefix+'/pic_12_E/pic_12_E',
               prefix+'/pic_24_HE/pic_24_HE',
               prefix+'/pic_48_HHE/pic_48_HHE']
    epoch = [30,14,16]
    batch_size = [2048,256,16]
    model_path = ['%s-%s'%(x,y) for x,y in zip(pre_dir,epoch)]
    #detectors
    detectors = [None,None,None]
    PNet = SingleDetector(P_Net,model_path[0])
    detectors[0] = PNet
    if pic_size == 48 or pic_size == 0:
        RNet = BatchDetector(R_Net,24,batch_size[1],model_path[1])
        detectors[1] = RNet
    if pic_size == 0:
        ONet = BatchDetector(O_Net,48,batch_size[2],model_path[2])
        detectors[2] = ONet 
    mtcnn_detector = MtcnnDetector(detectors, min_face_size, stride, thresh,scale_factor)
    
    run(wider_val_txt,wider_val_dir,output_file,detectors,mtcnn_detector)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#测试Onet
import numpy as np 
import os
import cv2
import sys
sys.path.append('../')
from path_and_config import path 
from assist.assist_train.forward import P_Net,R_Net,O_Net
from assist.assist_hard.mtcnndetect import MtcnnDetector
from assist.assist_hard.singledetector import SingleDetector
from assist.assist_hard.batchdetector import BatchDetector

def run(mtcnn_detector,video_path):
    video_capture = cv2.VideoCapture(video_path)
    video_capture.set(3,340)
    video_capture.set(4,480)
    cropbox = None
    while (video_capture.isOpened()):
        t1 = cv2.getTickCount()
        ret,frame = video_capture.read()
        if ret:
            frame = rotate(frame,90)#手机视频需要翻转
            image = np.array(frame)
            image = np.expand_dims(image,axis = 0)
            boxes_c,marks = mtcnn_detector.detect_face(image)
            t2 = cv2.getTickCount()
            t = (t2-t1)/cv2.getTickFrequency()
            fps = 1.0/t
            boxes_c = np.array(boxes_c)
            marks = np.array(marks)
            
            if (boxes_c.size) != 5: print('boxes size check');break
            #照片上绘box
            for i in range(boxes_c.shape[1]):
                #print('i',i)
                
                score = boxes_c[0,i,4]
                box = boxes_c[0,i,:4]
                cropbox = [int(box[0]),int(box[1]),int(box[2]),int(box[3])]
                cv2.rectangle(frame,(cropbox[0],cropbox[1]),(cropbox[2],cropbox[3]),(255,0,0),1)
                cv2.putText(frame,'{:.3f}'.format(score),(cropbox[0],cropbox[1]-2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2)
            cv2.putText(frame,'{:.4f}'.format(t)+ '' + '{:.3f}'.format(fps),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2)
            #照片上绘制mark
            #print(marks)
            for i in range(marks.shape[0]):
                print('i',i)
                print(len(marks[i]))
                for k in range(int(len(marks[i]))):
                    for j in range(int(len(marks[i][k])//2)):
                        print('j',j)
                        print(int(marks[i][k][2*j]))
                        print(int(marks[i][k][2*j+1]))
                        cv2.circle(frame,(int(marks[i][k][2*j]),int(marks[i][k][2*j+1])),3,(0,0,255),-1)
                   
            #        
            cv2.imshow('',frame)
            if cv2.waitKey(1)&0xFF == ord('q'):
                print('q stop')
                break
        else :
            print('device not find')
            break
    video_capture.release()
    cv2.destroyAllWindows()
def rotate(img, alpha):
    """
        given a face with bbox and landmark, rotate with alpha
        and return rotated face with bbox, landmark (absolute position)
    """
    h,w,c = img.shape
    center = (w/2, h/2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)#旋转的矩阵
    #whole image rotate
    #pay attention: 3rd param(col*row)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat,(img.shape[1],img.shape[0]))
    
    #crop face 
    face = img_rotated_by_alpha[0:h+1,0:h+1]
    return (face)       

                 
if __name__ == '__main__':
    pic_size = 0

    choice = ['_E','_HE','_HHE']
    if pic_size == 24: c = 0; print('we will test PNet')
    if pic_size == 48: c = 1; print('we will test RNet')
    if pic_size == 0:  c = 2; print('we will test ONet')
    pnp_pic_hard_sample = choice[c]  
    thresh = [0.3,0.1,0.7]
    min_face_size = 24
    stride = 2
    scale_factor = 0.79
#val图像、txt地址，输出图像地址
    #video_path = './video_test.avi'
    video_path = './qdw.mp4'
        
#model path
    prefix = path.root +'/train_model'
    pre_dir = [prefix+'/pic_12_E/pic_12_E',
               prefix+'/pic_24_HE/pic_24_HE',
               prefix+'/pic_48_HHE/pic_48_HHE']
    epoch = [18,14,10]
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
    run(mtcnn_detector,video_path)


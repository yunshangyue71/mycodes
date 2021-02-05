#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from assist.assist_hard.iter_data import TestLoader

def run(mtcnn_detector,test_data,gt_imdb):
    all_boxes, marks= mtcnn_detector.detect_face(test_data)
    count = 0  
    cc = 1
    print('marks\n', marks)
    print('all_boxes:\n',all_boxes)
    for imagepath in gt_imdb:
        print('\n',imagepath)
        image = cv2.imread(imagepath)
        print(image.shape)
        for box in all_boxes[count]:
           
            
            cv2.putText(image,str(np.round(box[4],2)),(int(box[0]),int(box[1])),
                        cv2.FONT_HERSHEY_TRIPLEX,1,color = (255,0,255))
            
            cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255))
            #print(marks)
            #print(len(marks[0]))
     
            
            for i in range(len(marks[count])):
                #print(len(marks[count][i]),'s')
                for j in range(len(marks[count][i])//2):
                   # print(i,j)
                    x  = int(marks[count][i][j*2])#* (box[2]-box[0])+box[0])
                    y =  int(marks[count][i][j*2+1])# * (box[3] - box[1])+box[1])
                    #print(marks[count][i][j*2],type(marks[count][i][j*2]))
                    #print(marks[count][i][j*2+1],type(marks[count][i][j*2+1]))
                    #print(x,type(x))
                    #print(y,type(y))
                    #print(type(cc),'aa')
                
                    cv2.circle(image, 
                           (x,y), 
                           4, 
                           (250,0,0),
                           -1
                           )
               
           
        count = count + 1
        cv2.imwrite('/media/q/deep/pictures/me/%d.jpg'%count,image)
        cv2.imshow('yushangyue',image)
        
        #cv2.imwrite(save_file,resized_im)
        cv2.waitKey(0)
    
        cv2.destroyAllWindows()
            
if __name__ == '__main__': 
    pic_size = 0

    choice = ['_E','_HE','_HHE']
    if pic_size == 24: c = 0; print('we will test PNet')
    if pic_size == 48: c = 1; print('we will test RNet')
    if pic_size == 0:  c = 2; print('we will test ONet')
    pnp_pic_hard_sample = choice[c]  
    thresh = [0.3,0.1 ,0.9 ]
    min_face_size = 20
    stride = 2
    scale_factor = 0.79
#val图像、txt地址，输出图像地址 
    gt_imdb = []
    #testdata_path = '/media/q/deep/mtcn_landmark_dataset_github/test/lfpw_testImage'
    testdata_path = '/media/q/deep/pictures/me'
    for item in os.listdir(testdata_path):
        gt_imdb.append(os.path.join(testdata_path,item))
        test_data = TestLoader(gt_imdb) 
        
#model path
    prefix = path.root +'/train_model'
    pre_dir = [prefix+'/pic_12_E/pic_12_E',
               prefix+'/pic_24_HE/pic_24_HE',
               prefix+'/pic_48_HHE/pic_48_HHE']
    epoch = [30,14,10]
    batch_size = [2048,256,16]
    model_path = ['%s-%s'%(x,y) for x,y in zip(pre_dir,epoch)]
    #model_path = ['./model/PNet/PNet-30','./model/RNet/RNet-22','./model/ONet/ONet-14']
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
    run(mtcnn_detector,test_data,gt_imdb)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
def run_5590mark122448():
    path = '/media/q/deep/mtcnn_brief/pic_txt/mark_48.txt'
    with open(path,'r') as f:
        datas = f.readlines()
        for data in datas:
            data = data.strip('\n').split(' ')
            print(data[0])
            img = cv2.imread(data[0])
            h,w,c = img.shape
            print(h,w,c)
            mark = data[2:]
            print(range(len(mark)//2))
            for i in range(len(mark)//2):
                x0 = float(mark[(i*2)])
                y0 = float(mark[(i*2+1)])
                x = w * x0
                y = h * y0
            
                cv2.circle(img,(int(x),int(y)),1,(250,0,0),-1)
            name=data[0].split('/')[-1]
            cv2.imshow(name,img)
         
            if cv2.waitKey(0)&0XFF == ord('z'):
                cv2.destroyAllWindows()   
                continue
            else:
                if cv2.waitKey(0)&0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
        cv2.destroyAllWindows()
                 
def run_5590mark():
    path = '../txt/trainImageList.txt'
    path_mark = '/media/q/deep/mtcn_landmark_dataset_github/train'
    with open(path, 'r') as f:
        infos = f.readlines()
        for info in infos:
            data = []
            info = info.strip('\n').split(' ')
            info[0]=info[0].replace('\\','/')
            info[0] = os.path.join(path_mark,info[0])
            data.append(info[0])
            
            for i in range(10):
                data.append(info[5+i])
            print(data)
            #break
            img = cv2.imread(data[0])
            h,w,c = img.shape
            print(h,w,c)
            mark = data[1:]
            print(range(len(mark)//2))
            for i in range(len(mark)//2):
                x0 = float(mark[(i*2)])
                y0 = float(mark[(i*2+1)])
                x = 1 * x0
                y = 1* y0
            
                cv2.circle(img,(int(x),int(y)),3,(250,0,0),-1)
            cv2.imshow(data[1],img)
    
    
            if cv2.waitKey(0)&0XFF == ord('z'):
                cv2.destroyAllWindows()
                continue
               
            else:
                if cv2.waitKey(0)&0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            

         
            
 
    
      
if __name__=='__main__':
    
    #mark12,24,48查看
    run_5590mark122448()
      
    #marktxt查看
    #run_5590mark()
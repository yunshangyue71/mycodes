# -*- coding: utf-8 -*-
'''
@time: 2019/01/11 11:28
spytensor
'''

import os
import json
import numpy as np
import pandas as pd
import glob
import cv2
import os
import shutil
from IPython import embed
from sklearn.model_selection import train_test_split
np.random.seed(41)


#0为背景
# classname_to_id = {"person": 1,
#                    "1":1, "2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9}

class Txt2CoCo:
    #image_dir: 存放照片的路径
    #total_annos:所有图片， 除了名字的annotation
    #classNameAndClassId:{"className0":int classId ,"background":"0",...}
    def __init__(self,image_dir,total_annos, classNameAndClassId):
        self.images = []        #存放所有的照片信息， 分辨率等等
        self.annotations = []   #存放所有的label 信息， bbox， label ， seg等等
        self.categories = []    #存放所有的分类信息，
        self.img_id = 0         #coco json中每个图片的id
        self.ann_id = 0         #coco json中每个annotation 的id， 一个图片可能有多个anno
        self.image_dir = image_dir
        self.total_annos = total_annos
        self.classNameAndClassId = classNameAndClassId

    # 由txt文件构建COCO
    # imgNames：所有的图片名字
    def to_coco(self, imgNames):
        self._init_categories()
        for imgName in imgNames:

            self.images.append(self._image(imgName))
            imgAnnos = self.total_annos[imgName]
            for imgAnno in imgAnnos:
                # print(imgAnno)
                bboxi = []
                for cor in imgAnno[:4]:
                    bboxi.append(int(cor))
                label = int(imgAnno[4])
                if str(label) not in self.classNameAndClassId:
                    continue
                annotation = self._annotation(bboxi,label)
                annotation["score"] = imgAnno[5]
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        #这个就是整个的dataset的annotation
        instance = {}
        instance['info'] = {"year":"","version":"",
                            "description":"","contributor":"",
                            "url":"","data_created":""} #这个一般用不到
        instance['license'] = {"id":"","name":"", "url":""}#这个一般也用不到
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in self.classNameAndClassId.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)
        print(self.categories)

    # 构建COCO的image字段
    def _image(self, imgName):
        image = {}
        img = cv2.imread(self.image_dir + imgName)
        #img = cv2.imread(path)
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = imgName
        return image

    # 构建COCO的annotation字段
    def _annotation(self, imgAnno,label):
        # label = shape[-1]
        points = imgAnno[:4]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        label = str(label)# TODO qdw
        annotation['category_id'] = int(self.classNameAndClassId[label])
        annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = self._get_area(points)
        return annotation

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return [min_x, min_y, max_x - min_x, max_y - min_y]
    # 计算面积
    def _get_area(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return (max_x - min_x+1) * (max_y - min_y+1)
    # segmentation
    def _get_seg(self, points):
        a = []
        b = points[4:]
        a.append(b)
        return a
    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示


if __name__ == '__main__':
    """需要提前设置的参数"""
    #txt 格式位置
    # xmin, ymin, w, h, cls
    # xmin, ymin, w, h, cls
    #coco格式 xmin，ymin，w， h

    #img copy or not
    imgCopyFlag = False

    # 该路径下存放所有的图片
    image_dir = "/media/q/data/datasets/VOC/VOC2007_test/JPEGImages/"

    #txt
    txtDir = "/media/q/data/datasets/VOC/VOC2007_test/format_me/Main/person/result/"

    #将整理好的coco 格式存放在什么位置
    saved_coco_path = "/media/q/data/datasets/VOC/VOC2007_test/format_me/Main/person/"

    # 这个是将整个数据集分为训练集和测试集比例
    testSize = 0.0

    # 整个数据集用不用打乱
    shuffle = False

    # classNameAndClassId:{"className0":int classId ,"background":"0",...}
    #"className0" 这个就是csv文件中的名字，就是这个类是哪个类
    classNameAndClassId = {"0" : 0, "1":1, "2":2,"3":3,
                           "4" : 4, "5":5,"6":6,
                           "7" : 7,"8":8,"9":9}
    """END"""


    # 整合csv格式标注文件
    # {imgName：[[anno0], [anno1]]}
    total_annotations = {}
    txtnames = os.listdir(txtDir)
    for i in range(len(txtnames)):
        info = np.loadtxt(txtDir + txtnames[i])
        info = info.reshape((-1, 6))
        imgname = txtnames[i].split('.')[0]+".jpg"
        info[:, 2:4] +=info[:,:2]
        total_annotations[imgname] = info

    #所有的图片名字
    imgNamesTotal = list(total_annotations.keys())

    #将图像名字为训练集测试集
    imgNamesTrain, imgNamesVal = train_test_split(imgNamesTotal, test_size=testSize, shuffle = shuffle)
    print("train_n:", len(imgNamesTrain), 'val_n:', len(imgNamesVal))

    # 创建必须的文件夹
    if not os.path.exists('%scoco/annotations/'%saved_coco_path):
        os.makedirs('%scoco/annotations/'%saved_coco_path)
    if not os.path.exists('%scoco/images/train2017/'%saved_coco_path):
        os.makedirs('%scoco/images/train2017/'%saved_coco_path)
    if not os.path.exists('%scoco/images/val2017/'%saved_coco_path):
        os.makedirs('%scoco/images/val2017/'%saved_coco_path)

    #copy images
    if imgCopyFlag:
        for file in imgNamesTrain:
            shutil.copy(image_dir+file,"%scoco/images/train2017/"%saved_coco_path)
        for file in imgNamesVal:
            shutil.copy(image_dir+file,"%scoco/images/val2017/"%saved_coco_path)

    # 把训练集转化为COCO的json格式
    l2c_train = Txt2CoCo(image_dir=image_dir,total_annos=total_annotations, classNameAndClassId=classNameAndClassId)
    train_instance = l2c_train.to_coco(imgNamesTrain)
    l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train20172.json'%saved_coco_path)

    # 把验证集转化为COCO的json格式
    l2c_val = Txt2CoCo(image_dir=image_dir,total_annos=total_annotations,  classNameAndClassId=classNameAndClassId)
    val_instance = l2c_val.to_coco(imgNamesVal)
    l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2017.json'%saved_coco_path)
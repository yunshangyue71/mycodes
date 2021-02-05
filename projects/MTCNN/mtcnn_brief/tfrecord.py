#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:51:01 2019

@author: q
"""
import os 
import sys
import random
import math
import tensorflow as tf
from datetime import datetime

from path_and_config import path
from assist.tfrecord_assist import data_tostring,to_tfexample
def path_init(pic_size,pnp_pic_hard_sample,individual,shuffling = True):
    tfrecord_folder = os.path.join(path.root,'tfrecord')
    if not os.path.exists(tfrecord_folder):
        os.mkdir(tfrecord_folder)
     
    txt_dir = os.path.join(path.root,'pic_txt/pic_%d%s.txt'%(pic_size,pnp_pic_hard_sample))
    tfrecord_dir_list = []
    for i in range(individual):
        tfrecord_dir = os.path.join(tfrecord_folder,'pic_%d%s%d.tfrecord'%(pic_size,pnp_pic_hard_sample,i))
        tfrecord_dir_list.append(tfrecord_dir)
    if shuffling:#打乱
        for i in range(individual):
            tfrecord_dir_list[i] = tfrecord_dir_list[i] +'_shuffle'  
    for tfrecord_dir in tfrecord_dir_list: 
        if os.path.exists(tfrecord_dir):
            print(tfrecord_dir,'tfrecord已经存在，不用重新创建，并退出')
            raise Exception
    print('tfrecord文件将要创建')
    return txt_dir,tfrecord_dir_list
def run(txt_dir,tfrecord_dir_list,individual, shuffling = True):
    dataset = get_dataset(txt_dir)
    print('%s   has been read'%(txt_dir))
    tfrecord_num = len(tfrecord_dir_list)
    assert individual == tfrecord_num 
    data_num = len(dataset)
    if shuffling:#打乱
        random.shuffle(dataset)
    datasets = []#将dataset拆分成individual个列表
    for i in range(tfrecord_num):
        start = math.ceil(i * data_num/(tfrecord_num))
        end =  math.ceil((i+1) * data_num/(tfrecord_num))
        datasets_ = dataset[start: end]
        datasets.append(datasets_)
    for j,data in enumerate(datasets):#构建individual个tfrecord
        print(datetime.now())
        with tf.python_io.TFRecordWriter(tfrecord_dir_list[j]) as tfrecord_writer:
            print(tfrecord_dir_list[j])
            for i, data_an_example in enumerate(data):
                data_dir = data_an_example['data_dir']
                tostring_data, height, width = data_tostring(data_dir)
                example = to_tfexample(data_an_example, tostring_data)
                tfrecord_writer.write(example.SerializeToString()) 
                if (i+1) % 100 == 0:
                    sys.stdout.write('\r>>%d/%d images hase been converted'% (i+1, len(data)))
                sys.stdout.flush()
        print(datetime.now(),'%d/%d tfrecord done'%(j+1,individual))
    
def get_dataset(txt_dir):
    dataset = []
    with open(txt_dir, 'r') as f:
        for line in f.readlines():
            inf = line.strip().split(' ')
            bbox = dict()
            bbox['xmin'] = 0; bbox['ymin'] = 0; bbox['xmax'] = 0; bbox['ymax'] = 0
            bbox['xlefteye'] = 0;     bbox['ylefteye'] = 0
            bbox['xrighteye'] = 0;    bbox['yrighteye'] = 0
            bbox['xnose'] = 0;        bbox['ynose'] = 0
            bbox['xleftmouth'] = 0;   bbox['yleftmouth'] = 0
            bbox['xrightmouth'] = 0;  bbox['yrightmouth'] = 0
            if len(inf) == 6:
                bbox['xmin'] = float(inf[2]); bbox['ymin'] = float(inf[3])
                bbox['xman'] = float(inf[4]); bbox['yman'] = float(inf[5])
            if len(inf) == 12:
                bbox['xlefteye'] = float(inf[2]);        bbox['ylefteye'] = float(inf[3])
                bbox['xrighteye'] = float(inf[4]);       bbox['yrighteye'] = float(inf[5])
                bbox['xnose'] = float(inf[6]);           bbox['ynose'] = float(inf[7])
                bbox['xleftmouth'] = float(inf[8]);      bbox['yleftmouth'] = float(inf[9])
                bbox['xrightmouth'] = float(inf[10]);    bbox['yrightmouth'] = float(inf[11])
            data_an_example = dict()
            data_an_example['data_dir'] = inf[0]
            data_an_example['label'] = int(inf[1])
            data_an_example['box_or_mark'] = bbox
            dataset.append(data_an_example)
    return dataset   
if __name__ == '__main__':
    pic_size = 48#第一次PNet
    #pic_size = 24#第二次RNet
    #pic_size = 48#第三次ONet
    choice = ['_E','_HE','_HHE']
    if pic_size == 12: c = 0; individual = 1#4将一个txt总共生成多少个tfrecord文件，1或者4个
    if pic_size == 24: c = 1; individual = 4
    if pic_size == 48: c = 2; individual = 4
    pnp_pic_hard_sample = choice[c]   
    shuffling = True
    
    txt_dir,tfrecord_dir_list = path_init(pic_size,pnp_pic_hard_sample,individual,shuffling = True)
    run(txt_dir,tfrecord_dir_list,individual, shuffling = True)
    
   

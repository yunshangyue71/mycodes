#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os 
import random
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import cv2

from path_and_config import path,config
from assist.assist_train.read_tfrecord import read_tfrecord
from assist.assist_train.forward import forward
from assist.assist_train.train_lr_op import train_lr_op

def train(tfrecord_dir_pre,  model_pre,logs_folder, pic_size, num, forward, end_epoch, display=100, base_lr=0.001):   
    
    """placeholder的声明"""
    input_image = tf.placeholder(tf.float32,  shape = [config.BATCH_SIZE, pic_size, pic_size, 3],
                                 name = 'input_image')
    label = tf.placeholder(tf.float32, shape = [config.BATCH_SIZE],name = 'label')
    box_target = tf.placeholder(tf.float32,shape=[config.BATCH_SIZE, 4], name = 'box_target')
    mark_target = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 10], name = 'mark_target')
    """forward op，train_op,lr_op结果是损失"""
    input_image = image_color_distort(input_image)
    cls_loss_op, box_loss_op,mark_loss_op, L2_loss_op,accuracy_op =\
        forward(pic_size,input_image, label, box_target, mark_target, training = True)#前向传播
    """误差权重确认,以及整体误差"""
    if pic_size == 12:    radio_cls_loss = 1.0;   radio_box_loss = 0.5;  radio_mark_loss = 0.5
    elif pic_size == 24:  radio_cls_loss = 1.0;   radio_box_loss = 0.5;  radio_mark_loss = 0.5
    elif pic_size == 48:  radio_cls_loss = 1.0;   radio_box_loss = 0.5;  radio_mark_loss = 1.0
    total_loss_op = (radio_cls_loss*cls_loss_op +  radio_box_loss*box_loss_op +
                     radio_mark_loss*mark_loss_op+ L2_loss_op)
    """对网络训练的操作，以及对学习率改变的操作"""
    train_op, lr_op = train_lr_op(base_lr, total_loss_op, num)
        
    """tensorboard中记录的标量"""
    tf.summary.scalar('cls_loss', cls_loss_op)
    tf.summary.scalar('box_loss', box_loss_op)
    tf.summary.scalar('landmark_loss',mark_loss_op)
    tf.summary.scalar('cls_accuracy', accuracy_op)
    tf.summary.scalar('total_loss', total_loss_op)
    summary_op = tf.summary.merge_all()
    """tfrecord文件路径指定， 训练数据集从tfrecord读出"""
    image_batch, label_batch, box_batch, mark_batch =   read_tfrecord(tfrecord_dir_pre, config.BATCH_SIZE,pic_size)
    print('############################################################')
    """session 初始化"""
    init = tf.global_variables_initializer()
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep = 0)
    sess.run(init)
    
    writer = tf.summary.FileWriter(logs_folder, sess.graph)
    projector_config = projector.ProjectorConfig()
    projector.visualize_embeddings(writer, projector_config)
    """协同管理"""
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    
    batch_done_count = 0#网络计算了批量数,每个epoch都重新计数
    MAX_STEP = int(num/config.BATCH_SIZE + 1) * end_epoch
    epoch_count = 0
    try :
        for step in range(MAX_STEP):
            batch_done_count +=1
            if coord.should_stop():
                break
            image_batch_array, label_batch_array,box_batch_array,mark_batch_array = \
            sess.run([image_batch, label_batch,box_batch,mark_batch])
            #随机旋转
            image_batch_array,mark_batch_array = \
            random_flip_images(image_batch_array, label_batch_array, mark_batch_array)
            _,_,summary = sess.run([train_op, lr_op, summary_op],
                                   feed_dict={input_image:image_batch_array, 
                                              label:label_batch_array,
                                              box_target:box_batch_array,
                                              mark_target:mark_batch_array})
            writer.add_summary(summary, global_step = step)
            """满100就展示"""
            if (step +1 ) % (display) ==0:
                cls_loss, box_loss, mark_loss,L2_loss, lr, acc = sess.run(
                        [cls_loss_op, box_loss_op,mark_loss_op, L2_loss_op, lr_op, accuracy_op],
                         feed_dict = {input_image:image_batch_array,
                                      label:label_batch_array,
                                      box_target:box_batch_array,
                                      mark_target:mark_batch_array})
                total_loss = radio_cls_loss * cls_loss + radio_box_loss * box_loss +\
                             radio_mark_loss * mark_loss + L2_loss      
                print("Step:%d/%d,  acc:%3f,  cls loss:%4f, box_loss:%4f,  mark loss:%4f,  L2 loss: %4f,  Total Loss:%4f,  lr:%f"
                      %( step+1,MAX_STEP, acc, cls_loss, box_loss,mark_loss, L2_loss, total_loss, lr))
            """下一轮迭代"""
            if batch_done_count * config.BATCH_SIZE > num*2:
                epoch_count = epoch_count + 1
                batch_done_count = 0
                path_prefix = saver.save(sess, model_pre, global_step = epoch_count * 2)
                print('path prefix is :', path_prefix)

    except tf.errors.OutOfRangeError:
        print("训练完成！！！")
    finally:
        coord.request_stop()
        writer.close()
    coord.join(threads)
    sess.close()
def random_flip_images(image_batch, label_batch, mark_batch):
    if random.choice([0, 1]) > 0:
        #num_images = image_batch.shape[0]
        flip_mark_indexes = np.where(label_batch == -2)[0]
        flip_pos_indexes = np.where(label_batch == 1)[0]
        
        flip_indexes = np.concatenate((flip_mark_indexes, flip_pos_indexes))
        for i in flip_indexes:
            cv2.flip(image_batch[i], 1, image_batch[i])
        for i in flip_mark_indexes:
            mark_ = mark_batch[i].reshape((-1, 2))
            mark_ = np.asarray([(1-x,y) for (x,y) in mark_])
            l_copy = mark_
            """
            mark_[0] = l_copy[1]
            mark_[1] = l_copy[0]
            mark_[3] = l_copy[4]
            mark_[4] = l_copy[3]
            """
            mark_[[0,1]] = l_copy[[1,0]]
            mark_[[3,4]] = l_copy[[4,3]]
            #generator怎么回事
            mark_batch[i] = mark_.ravel()
    return image_batch, mark_batch


"""
输入图像的对比度，亮度， 色相，饱和度的随机调节
"""
def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower = 0.5,upper = 1.5)
    #随机调整对比度范围0.5-1.5
    inputs = tf.image.random_brightness(inputs, max_delta = 0.2)
    #随机调整图像的亮度（0-0.2）
    inputs = tf.image.random_hue(inputs, max_delta = 0.2)
    #随机调整色调-0.2，0.2，max_delta <0.5
    inputs = tf.image.random_saturation(inputs, lower = 0.5, upper = 1.5)
    #随机调节饱和度，
    return inputs

    
if __name__ == '__main__': 
    """使用前必须指定""" 
    pic_size = 48#第一次PNet
    #pic_size = 24#第二次RNet
    #pic_size = 48#第三次ONet
    choice = ['_E','_HE','_HHE']
    if pic_size == 12: c = 0;net = 'PNet'
    if pic_size == 24: c = 1;net = 'RNet'
    if pic_size == 48: c = 2;net = 'ONet'
    pnp_pic_hard_sample = choice[c]                
    
    end_epoch = 14
    display = 100
    lr = 0.001
    HS = pnp_pic_hard_sample#名字太长
    """txt,tfrecord,model,logs文件夹的指定"""   
    with open(os.path.join(path.root,'pic_txt/pic_%d%s.txt'%(pic_size,pnp_pic_hard_sample)),'r') as f:
        dataset = f.readlines()
        num = len(dataset)
    tfrecord_folder = os.path.join(path.root,'tfrecord')
    model_folder = os.path.join(path.root,'train_model/pic_%d%s'%(pic_size,pnp_pic_hard_sample))
    if not os.path.exists(model_folder):  os.mkdir(model_folder)
    tfrecord_dir_pre = '%s/pic_%d%s'%(tfrecord_folder,pic_size,pnp_pic_hard_sample)
    logs_folder = os.path.join(path.root,'logs/%s'%net)
    if not os.path.exists(logs_folder): os.mkdir(logs_folder)
    model_pre = os.path.join(model_folder,'pic_%d%s'%(pic_size,pnp_pic_hard_sample))
    
    if pic_size == 12: print('pic_size is %d,train net is PNet.'%(pic_size))
    if pic_size == 24: print('pic_size is %d,train net is RNet.'%(pic_size))
    if pic_size == 48: print('pic_size is %d,train net is ONet.'%(pic_size))
    print('The tfrecord pre is ',tfrecord_dir_pre)
    print('model_folder： ', model_folder) 
    print('log folder: ', logs_folder)
    print('The number of train data is ',num)
    print('batch size:',config.BATCH_SIZE,'. end_epoch:',end_epoch)
    print('###########################################')
    
    tf.reset_default_graph()
    train(tfrecord_dir_pre, model_pre,logs_folder, pic_size, num,forward, end_epoch, display=display, base_lr=lr)
    
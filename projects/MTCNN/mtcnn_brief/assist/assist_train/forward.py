#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np


num_keep_ratio = 0.7

def forward(pic_size,input_image, label, box_target, mark_target,training = True):
    if pic_size == 12:
        cls_loss, box_loss, mark_loss, L2_loss, accuracy =\
        P_Net(input_image, label = label, box_target=box_target, mark_target = mark_target,training = True)
    if pic_size == 24:
        cls_loss, box_loss, mark_loss, L2_loss, accuracy =\
        R_Net(input_image, label = label, box_target=box_target, mark_target=mark_target,training=True)
    if pic_size ==48:
        cls_loss, box_loss, mark_loss, L2_loss, accuracy =\
        O_Net(input_image, label = label, box_target=box_target, mark_target=mark_target,training=True)
    return cls_loss, box_loss, mark_loss, L2_loss, accuracy
"""
inputs：输入一个tf量
return pos + 0.25 * neg
"""
def prelu(inputs):
    alphas = tf.get_variable('alphas', 
                             shape = inputs.get_shape()[-1],
                             dtype = tf.float32,
                             initializer = tf.constant_initializer(0.25))
    #创建一个tf的变量
    
    """将正负值分别取出"""
    pos = tf.nn.relu(inputs)
    #inputs正值保留，负值为0
    neg = (inputs-abs(inputs)) * 0.5
    #inputs负值保留，正值为0
    return pos + alphas * neg
    #起作用的不仅仅是正值，负值也起作用，不过作用权值为0.25
    
    

"""
cls_prob:(BATCH_SIZE,2)
label:(BATCH_SIZE,)
return 人脸分类的精度
"""
def cls_ohem(cls_prob, label):
    zeros = tf.zeros_like(label)
    label_pick_pos = tf.where(tf.less(label, 0),zeros, label)
    #mark-2,part-1以及neg0统统都看做是0?？?
    num_cls_prob = tf.size(cls_prob)#B*2
    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob, -1])#(B*2,)
    label_pick_pos_int = tf.cast(label_pick_pos, tf.int32)
    num_row = tf.to_int32(cls_prob.get_shape()[0])#B
    row = tf.range(num_row) * 2
    #[0,2,4,......B*2-2]
    indices_ = row + label_pick_pos_int#(B,2)中第一个是不是人脸的概率，第二是人脸概率
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))#选择出了是人脸的人脸概率和不是人脸的费人脸概率
    loss = -tf.log(label_prob + 1e-10)
    """ 计算neg，pos的个数   """
    zeros = tf.zeros_like(label_prob, dtype = tf.float32)
    ones = tf.ones_like(label_prob, dtype = tf.float32)
    valid_inds = tf.where(label<zeros, zeros, ones)#part,mark就不算了
    num_valid = tf.reduce_sum(valid_inds)
    #只计算pos中的正确率，讲neg，mark，part中对于精度的影响去掉了
    loss = loss * valid_inds
    keep_num = tf.cast(num_valid * num_keep_ratio, dtype = tf.int32)
    loss, _ = tf.nn.top_k(loss, k = keep_num)
    loss = tf.reduce_mean(loss)
    return loss
                    
    

"""
box_target:标定的label框
box_pred:预测的label框
label：标签
return:返回框的loss

"""
def box_ohem(box_pred, box_target, label):
    zeros_index = tf.zeros_like(label, dtype = tf.float32)
    ones_index = tf.ones_like(label, dtype=tf.float32)
    valid_inds = tf.where(tf.equal(tf.abs(label), 1), ones_index, zeros_index)
    #pos,part置位为1.neg,mark为0
    square_error = tf.square(box_pred - box_target)
    #这个也就是方框四个值的offset的平方
    square_error = tf.reduce_sum(square_error, axis = 1)
    #[x1的所有offset的平方和y1,x2,y2]
    num_valid = tf.reduce_sum(valid_inds)
    #统计pos，part的个数
    keep_num = tf.cast(num_valid, dtype = tf.int32)
    square_error = square_error * valid_inds#去除neg，mark关于box loss的贡献
    _, k_index = tf.nn.top_k(square_error, k = keep_num)#这一步其实是全部选择了。
    square_error = tf.gather(square_error, k_index)
    #选出最大值位置处的数值
    return tf.reduce_mean(square_error)  
    
"""
label:应该是生成txt的时候放进去的-1， -2，这个就在图片名字后面
"""
def mark_ohem(mark_pred, mark_target, label):
    ones = tf.ones_like(label, dtype = tf.float32)
    zeros = tf.zeros_like(label, dtype = tf.float32)
    valid_inds = tf.where(tf.equal(label, -2), ones, zeros)
    #选中mark
    square_error = tf.square(mark_pred - mark_target)
    square_error = tf.reduce_sum(square_error, axis = 1)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid*num_keep_ratio, dtype = tf.int32)
    square_error = square_error * valid_inds
    
    #square_error = tf.reduce_sum(square_error)/num_valid
    #return square_error
    _, k_index = tf.nn.top_k(square_error, k = keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)
   


"""
cls_prob:人脸分类
label:标签
"""

def cal_accuracy(cls_prob, label):
    pred = tf.argmax(cls_prob, axis = 1)
    label_int = tf.cast(label, tf.int64)
    cond = tf.where(tf.greater_equal(label_int, 0))#label中等于0的位置
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int, picked)#挑选label中neg组成的列表，内容其实为0
    pred_picked = tf.gather(pred, picked)#挑选label中是neg的位置中，对应的cls——prob那个值比较大
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked, pred_picked), tf.float32))#求处所有费人脸狂的的平均值
    return accuracy_op
    

"""
在tb中添加直方图

"""
def _activation_summary(x):
    tensor_name = x.op.name
    #获取x计算图中的名字
    print('load summary for :', tensor_name)
    tf.summary.histogram(tensor_name + '/activations', x)
    #在计算图中添加x/activations的张量，值为x
    
    

"""
inputs:输入图像，矩阵的格式是tf.float32形式的
label：归属标签，tf.float32
box_target:label框
mark_target:关键点信息
training：判断是否是进行的训练还是测试什么的
返回的前向传播的结果
"""
def P_Net(inputs, label = None, box_target=None,
          mark_target = None,training = True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer = slim.xavier_initializer(),
                        biases_initializer =  tf.zeros_initializer(),
                        weights_regularizer = slim.l2_regularizer(0.0005),
                        padding = 'VALID',
                        ):
        print('input_image的形状：',inputs.get_shape())
        if training:
            print('label shape',label.get_shape())
            print('box target',box_target.get_shape())
            print('mark target',mark_target.get_shape())
        net= slim.conv2d(inputs, 10, 3, stride = 1, scope = 'conv1')
        _activation_summary(net)
        print('P卷积1层后：',net.get_shape())
        net = slim.max_pool2d(net, kernel_size = [2, 2], stride = 2, scope = 'pool1', padding = 'SAME')
        _activation_summary(net)
        print(net.get_shape())
        net = slim.conv2d(net, num_outputs = 16, kernel_size = [3, 3], stride = 1, scope = 'conv2')
        _activation_summary(net)
        print('P卷积2层后：',net.get_shape())#卷几2层
        net = slim.conv2d(net, num_outputs = 32, kernel_size = [3, 3], stride = 1, scope = 'conv3')
        _activation_summary(net)
        print('P卷积3层后：',net.get_shape())#卷积3层
             
        conv4_1 = slim.conv2d(net, num_outputs = 2, kernel_size = [1, 1],stride = 1,scope = 'conv4_1',activation_fn = tf.nn.softmax)
        _activation_summary(conv4_1)
        print('Pcls prob:',conv4_1.get_shape())#(B，1,1,2)
        box_pred = slim.conv2d(net, num_outputs = 4, kernel_size = [1, 1], stride = 1, scope = 'conv4_2', activation_fn = None)
        _activation_summary(box_pred)
        print('Pbox pred',box_pred.get_shape())#(B，1,1,4)
        mark_pred = slim.conv2d(net, num_outputs = 10, kernel_size = [1, 1], stride = 1, scope = 'conv4_3', activation_fn = None)
        _activation_summary(mark_pred)
        print('Pmark pred',mark_pred.get_shape())#(B，1,1,10)
        if training:
            cls_prob = tf.squeeze(conv4_1, [1, 2], name = 'cls_prob')
            print('Pcls prob into ohem:',cls_prob.get_shape())#(B，2)
            cls_loss = cls_ohem(cls_prob, label)#只计算neg，pos中损失最大的前70%
            box_pred = tf.squeeze(box_pred, [1, 2], name = 'box_pred')
            print('Pbox prob into ohem:',box_pred.get_shape())#(B，4)
            box_loss = box_ohem(box_pred, box_target, label)
            mark_pred = tf.squeeze(mark_pred, [1,2], name = 'mark_pred')
            print('Pmark prob into ohem:',mark_pred.get_shape())#(B，10)
            mark_loss = mark_ohem(mark_pred, mark_target, label)
            accuracy = cal_accuracy(cls_prob,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss, box_loss, mark_loss, L2_loss, accuracy
        else :
            cls_pro_test = tf.squeeze(conv4_1, axis=0)
            box_pred_test = tf.squeeze(box_pred, axis=0)
            mark_pred_test = tf.squeeze(mark_pred, axis=0)          
            return cls_pro_test, box_pred_test, mark_pred_test
def R_Net(inputs,label = None, box_target=None,mark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer = tf.zeros_initializer(),
                        weights_regularizer = slim.l2_regularizer(0.0005),
                        padding='valid'):
        print('Rinput:',inputs.get_shape())
        net = slim.conv2d(inputs,num_outputs=28,kernel_size=[3,3],stride=1,scope='conv1')
        print('R卷积1层后:',net.get_shape())
        net = slim.max_pool2d(net,kernel_size = [3,3],stride = 2,scope = 'pool1',padding = 'SAME')
        print('Rpool1后:',net.get_shape())
        net = slim.conv2d(net,num_outputs=48,kernel_size=[3,3],stride=1,scope='conv2')
        print('R卷积2层后:',net.get_shape())
        net = slim.max_pool2d(net,kernel_size = [3,3],stride = 2,scope = 'pool2',padding = 'SAME')
        print('Rpool2后:',net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[2,2],stride=1,scope='conv3')
        print('R卷积3层后:',net.get_shape())
        fc_flatten = slim.flatten(net)
        print('Rflatten:',fc_flatten.get_shape())
        fc1 = slim.fully_connected(fc_flatten,num_outputs = 128, scope = 'fc1')
        print('R fc1:',fc1.get_shape())
        cls_prob = slim.fully_connected(fc1,num_outputs = 2,scope='cls_fc',activation_fn = tf.nn.softmax)
        print('R cls prob:',cls_prob.get_shape(),'cls')
        box_pred = slim.fully_connected(fc1,num_outputs = 4,scope = 'box_fc',activation_fn = None)
        print('R box pred:',box_pred.get_shape())
        mark_pred = slim.fully_connected(fc1,num_outputs = 10,scope ='mark_fc',activation_fn = None)
        print('R mark pred:',mark_pred.get_shape())
        if training:
            cls_loss=cls_ohem(cls_prob,label)
            box_loss = box_ohem(box_pred,box_target,label)
            mark_loss = mark_ohem(mark_pred,mark_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,box_loss,mark_loss,L2_loss,accuracy
        else:
            return cls_prob,box_pred,mark_pred
def O_Net(inputs,label = None,box_target=None,mark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer = tf.zeros_initializer(),
                        weights_regularizer = slim.l2_regularizer(0.0005),
                        padding = 'valid'):
        print('O input:',inputs.get_shape())
        net = slim.conv2d(inputs, num_outputs = 32,kernel_size = [3,3], stride = 1,scope='conv1')
        print('O conv1:',net.get_shape())
        net = slim.max_pool2d(net,kernel_size = [3,3], stride = 2, scope = 'pool1',padding = 'SAME')
        print('O pool1:',net.get_shape())
        net = slim.conv2d(net, num_outputs = 64, kernel_size = [3,3], stride = 1, scope = 'conv2')
        print('O conv2:',net.get_shape())
        net = slim.max_pool2d(net,kernel_size = [3,3], stride = 2, scope = 'pool2')
        print('O poll2:',net.get_shape())
        net = slim.conv2d(net,num_outputs = 64, kernel_size = [3,3], stride = 1, scope='conv3')
        print('O conv3:',net.get_shape())
        net = slim.max_pool2d(net, kernel_size = [2,2],stride = 2, scope = 'pool3', padding = 'SAME')
        print('O pool3:',net.get_shape())
        net = slim.conv2d(net, num_outputs = 128,kernel_size = [2,2], stride = 1, scope = 'conv4')
        print('O conv4:',net.get_shape())
        fc_flatten=slim.flatten(net)
        print('O flatten:',fc_flatten.get_shape())
        fc1 = slim.fully_connected(fc_flatten, num_outputs = 256, scope = 'fc1')
        print('O fc1:',fc1.get_shape())
        
        cls_prob = slim.fully_connected(fc1, num_outputs = 2,scope = 'cls_fc', activation_fn = tf.nn.softmax)
        print('O cls prob:',cls_prob.get_shape())
        box_pred = slim.fully_connected(fc1, num_outputs = 4, scope = 'box_fc', activation_fn = None)
        print('O box pred:',box_pred.get_shape())
        mark_pred = slim.fully_connected(fc1,num_outputs = 10, scope = 'mark_fc', activation_fn = None)
        print('O mark pred:',mark_pred.get_shape())
        if training:
            cls_loss = cls_ohem(cls_prob, label)
            box_loss = box_ohem(box_pred, box_target, label)
            accuracy = cal_accuracy(cls_prob, label)
            mark_loss = mark_ohem(mark_pred, mark_target, label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            #L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            return cls_loss, box_loss, mark_loss, L2_loss,accuracy
        else:
            return cls_prob, box_pred, mark_pred
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
from path_and_config import config

def train_lr_op(base_lr, loss, data_num):
    lr_factor = 0.1
    global_step = tf.Variable(0, trainable = False)
    
    boundaries = [int(epoch * data_num / config.BATCH_SIZE) \
                  for epoch in config.LR_EPOCH]
    lr_values = [base_lr * (lr_factor ** x) \
                 for x in range(0, len(config.LR_EPOCH) + 1)]
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    #根据global step所处的阶段的不同，对应不同的lr value
    optimizer = tf.train.MomentumOptimizer(lr_op, 0.9, name = 'Momentum')
    train_op = optimizer.minimize(loss, global_step)
    return train_op, lr_op
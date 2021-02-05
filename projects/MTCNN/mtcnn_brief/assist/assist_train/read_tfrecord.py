#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取tfrecord文件中的example放到内存队列
"""

import tensorflow as tf
import os
def read_tfrecord(tfrecord_dir_pre, batch_size,pic_size):
    """统计tfrecord个数,制成列表"""
    tfrecord_num = 0
    tfrecord_dir_list = []
    while True:
        tfrecord_dir = '%s%d.tfrecord_shuffle'%(tfrecord_dir_pre, tfrecord_num)
        if os.path.exists(tfrecord_dir):
            tfrecord_dir_list.append(tfrecord_dir)
        else:
            print('tfrecord is',range(tfrecord_num))
            break
        tfrecord_num += 1
    """构建文件队列"""
    filename_queue = tf.train.string_input_producer(tfrecord_dir_list, shuffle = True)   
    """从文件队列读出，一个例子并反序列化"""
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    image_features = tf.parse_single_example(
            serialized_example,
            features = {
                    'image/encoded':tf.FixedLenFeature([],tf.string),
                    'image/label':tf.FixedLenFeature([], tf.int64),
                    'image/roi':tf.FixedLenFeature([4], tf.float32),
                    'image/landmark':tf.FixedLenFeature([10],tf.float32)
                    })
    
    """放入内存队列，训练的时候可以直接从这里获取"""
    image = tf.decode_raw(image_features['image/encoded'],tf.uint8)
    image = tf.reshape(image,[pic_size,pic_size, 3])
    image = (tf.cast(image, tf.float32)-127.5)/128
    label = tf.cast(image_features['image/label'], tf.float32)
    roi = tf.cast(image_features['image/roi'],tf.float32)
    mark = tf.cast(image_features['image/landmark'],tf.float32)
    
    image, label, roi ,mark = tf.train.batch(
            [image, label, roi, mark],
            batch_size = batch_size,#从队列中获取的出队列的数量
            num_threads = 2,#入队线程的限制
            capacity = 1 * batch_size#设置队列的最大数量
            )
    label = tf.reshape(label,[batch_size])
    roi = tf.reshape(roi, [batch_size, 4])
    mark = tf.reshape(mark, [batch_size, 10])
    return image, label, roi, mark



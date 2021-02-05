#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
这个文件作用是，tfrecord文件中的一些辅助的函数
"""
import tensorflow as tf
import cv2
"""读取tostring的图片、宽、高"""
def data_tostring(a_data_dir):
    image = cv2.imread(a_data_dir)
    tostring_data = image.tostring()
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return tostring_data, height, width

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list = tf.train.Int64List(value = value))

def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = value))
def to_tfexample(data_an_example, image_buffer):
    class_label = data_an_example['label']
    box_or_mark = data_an_example['box_or_mark']
    roi = [box_or_mark['xmin'],box_or_mark['ymin'],box_or_mark['xmax'],box_or_mark['ymax']]
    landmark = [box_or_mark['xlefteye'],     box_or_mark['ylefteye'],
                box_or_mark['xrighteye'],    box_or_mark['yrighteye'],
                box_or_mark['xnose'],        box_or_mark['ynose'],
                box_or_mark['xleftmouth'],   box_or_mark['yleftmouth'],
                box_or_mark['xrightmouth'],  box_or_mark['yrightmouth']]
    #读取image_example的内容信息。
    example = tf.train.Example(features = tf.train.Features(feature={
                                                                        'image/encoded':_bytes_feature(image_buffer),
                                                                        'image/label':_int64_feature(class_label),
                                                                        'image/roi':_float_feature(roi),
                                                                        'image/landmark':_float_feature(landmark)
                                                                    }))
    return example

    
"""
老师您好：
    我有下面几个问题：
        1、如何定义一个迭代器类和一个生成器类
        2、迭代器类和生成器类是怎么工作的
                                ————谢谢老师
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2

"""这个载入的是一个迭代器，每个是一张照片"""
class TestLoader:
    def __init__(self, imdb, batch_size = 1, shuffle =False):
        self.imdb = imdb
        self.batch_size = batch_size#这个应该是每间隔多少去一个图片
        self.shuffle = shuffle
        self.size = len(imdb)
        self.cur = 0
        self.data = None
        self.label = None
        """执行的方法"""
        self.reset()#指针归0，将输入的图片的位置列表进行打乱
        self.get_batch()    
    """指针归0，将输入的图片的位置列表进行打乱"""
    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.imdb)
    """外面的iter这个函数以及内部的__iter__这个方法返回的是一个可迭代对象"""
    def __iter__(self):
        return self
    """执行next函数"""
    def __next__(self):
        return self.next()
    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data
        else:
            raise StopIteration
    """当前指针+batch_size是否会超过数据集的总数"""       
    def iter_next(self):
        return self.cur + self.batch_size <= self.size
    """获取cur处的一个图片"""
    def get_batch(self):
        imdb = self.imdb[self.cur]
        im = cv2.imread(imdb)
        self.data = im
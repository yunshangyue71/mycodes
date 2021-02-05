#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
class BatchDetector(object):
    def __init__(self, net_factory, data_size, batch_size, model_path):
        """构建计算图，载入模型"""
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[batch_size, data_size, data_size, 3], name='input_image')
            self.cls_prob, self.box_pred, self.mark_pred = net_factory(self.image_op, training=False)
            self.sess = tf.Session( config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print('model path: ',model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "最新的model存在"
            print("载入modelB")
            saver.restore(self.sess, model_path)

        self.data_size = data_size
        self.batch_size = batch_size
    def predict(self, databatch):
        batch_size = self.batch_size
        minibatch = []
        cur = 0
        n = databatch.shape[0]
        while cur < n:
            minibatch.append(databatch[cur:min(cur + batch_size, n), :, :, :])
            cur += batch_size
        cls_prob_list = []
        box_pred_list = []
        mark_pred_list = []
        for idx, data in enumerate(minibatch):
            m = data.shape[0]
            real_size = self.batch_size
            """最后一组，不够batch_size，就将本组的复制"""
            if m < batch_size:
                keep_inds = np.arange(m)
                gap = self.batch_size - m
                while gap >= len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m
                assert len(data) == batch_size
            """执行计算图操作"""
            cls_prob, box_pred,mark_pred = self.sess.run([self.cls_prob, self.box_pred,self.mark_pred], 
                                                              feed_dict={self.image_op: data})
            cls_prob_list.append(cls_prob[:real_size])
            box_pred_list.append(box_pred[:real_size])
            mark_pred_list.append(mark_pred[:real_size])
        #print('mark pred d \n',mark_pred_list[:10])
        return np.concatenate(cls_prob_list, axis=0), np.concatenate(box_pred_list, axis=0), np.concatenate(mark_pred_list, axis=0)

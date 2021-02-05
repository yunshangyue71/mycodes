#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 06:14:33 2019

@author: q
"""
from easydict import EasyDict as edict
path = edict()
path.WIDER_FACE = '/media/q/deep/wider_face/WIDER_train/images'
path.LFW5590 = '/media/q/deep/mtcn_landmark_dataset_github/train'
path.RESULTS_ROOT = '/media/q/deep/mtcnn_brief'
path.WIDER_VAL='/media/q/deep/wider_face/WIDER_val/images'
#重命名
path.pnp = path.WIDER_FACE 
path.mark = path.LFW5590  
path.root = path.RESULTS_ROOT 

config = edict()
config.BATCH_SIZE = 384
config.CLS_OHEM = True
config.BOX_OHEM = False
config.CLS_OHEM_RATIO = 0.7
config.BOX_OHEM_RATIO = 0.7

config.EPS = 1e-14
config.LR_EPOCH = [6, 14, 20]

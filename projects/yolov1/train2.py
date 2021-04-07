from data.dataloader_detection import ListDataset
from data.collate_function import collate_function
from config.config import load_config, cfg
from net.resnet import ResNet, ResnetBasic, ResnetBasicSlim
from net.yolov1 import YOLOv1
from loss.yololoss import yoloLoss
from dataY.yolov1_dataY import DataY
from loss.L1L2loss import Regularization

from torch.utils.data import SubsetRandomSampler
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
import time

if __name__ == '__main__':
    """config"""
    cfgpath = "./config/config.yaml"
    load_config(cfg, cfgpath)
    for key,value in cfg.items():
        for k, v in value.items():
            try:
                print(str(key) + "/" + str(k), str(v))
            except:
                print("")
    # print(cfg)
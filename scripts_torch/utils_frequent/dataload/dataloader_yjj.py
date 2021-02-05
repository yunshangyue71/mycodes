#dataloader_example
import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2 as cv
import json
from numpy.random import randint as npr
from aug import aug_image as ia


# from aug import crop_pad_rotate


class ListDataset(Dataset):
    def __init__(self, ):

        self.trainPath = 'D:/project/jersey/'
        self.imgpath = 'D:/data/street/train/'
        with open(self.trainPath + 'trainNumsAnnosOut.json', 'r') as f:
            self.annos = json.load(f)

        self.imgsize = [128, 128]
        self.mapsize = [32, 32]
        self.ratio = self.imgsize[0] / self.imgsize[1]
        self.flag = False

        self.ancor_boxes = np.zeros([4, self.mapsize[1], self.mapsize[0]])
        for i in range(self.mapsize[1]):
            for j in range(self.mapsize[0]):
                self.ancor_boxes[0, i, j] = float(j)
                self.ancor_boxes[1, i, j] = float(i)
                self.ancor_boxes[2, i, j] = float(1.0)
                self.ancor_boxes[3, i, j] = float(1.0)

    def __getitem__(self, index):
        img_path = self.imgpath + self.annos[index][0]
        img = cv.imread(img_path)
        h, w, _ = img.shape

        # Extract image as PyTorch tensor
        boxes = np.array(self.annos[index][1]).reshape(-1, 5).astype(np.float32).copy()
        ids = boxes[:, 0].astype(np.int)
        ids[ids == 10] = 0
        boxes = boxes[:, 1:]
        h, w, _ = img.shape

        cut, boxes = self.get_pieces(img, boxes)
        # cut, boxes = crop_pad_rotate(img, boxes)

        h, w, _ = cut.shape

        boxes[:, :2] += boxes[:, 2:] / 2

        boxes[:, 0] /= w
        boxes[:, 1] /= h
        boxes[:, 2] /= w
        boxes[:, 3] /= h
        cut = cv.resize(cut, (self.imgsize[0], self.imgsize[1]))
        cut = ia(cut)#读取数据以及数据增强

        if self.flag:
            print('这里绘图')

        else:
            cut = self.randomcut(img)
            cut = cv.resize(cut, tuple(self.imgsize))
        targets = np.concatenate([hm, delta_boxes, classes], axis=0)
        return torch.from_numpy(cut.transpose(2, 0, 1).astype(np.float32)), \
               torch.from_numpy(targets.astype(np.float32))

    def __len__(self):
        return len(self.annos)

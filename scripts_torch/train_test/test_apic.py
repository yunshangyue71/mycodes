import warnings
warnings.filterwarnings("ignore")
import torch
import cv2
import numpy as np
import os
import json
from model import Net


# 训练配置
# **********************************************************
network = Net()
# netHg = nn.DataParallel(network, devices = [0, 1, 2]) # 并行训练
network.load_state_dict(torch.load('model/cifar10_20.pt'))
# torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')
network.to(device)
network.eval()  # 测试的时候用
# ---------------------------------------------------------
if __name__ == '__main__':

    
    imageSize = [32, 32]
    imgsNp = np.zeros((1, 3, imageSize[0], imageSize[1]))
    flag = int(input("请确保输入的尺寸和模型训练时候的尺寸一致，请确保加载的网络、加载的权重是所需要的！ 1：表示确认"))
    if flag != 1: 
        raise AssertionError
    
    root = 'D:\\project\\cifar10\\cifar_test\\'
    with open('./cifar10_test.json', 'r') as f:
        annoDic = json.load(f)
    annoDic = np.array(annoDic)
    imgDirList = annoDic[:, 0]
    labels = annoDic[:, 1]
    labelDic = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'trunk']
    rightNumTest = 0
    for i in range(len(imgDirList)):
        img = cv2.imread(root + imgDirList[i])
        img_ = cv2.resize(img, (imageSize[0], imageSize[1]))
        img = img.transpose(2, 0, 1).astype(np.float) / 255.0

        imgsNp[0] = img

        imgs = torch.from_numpy(imgsNp).float()
        imgs = imgs.to(device).float()
        with torch.no_grad():
            preds = network(imgs).reshape((-1, 10))
            preds = preds.detach().cpu().numpy()
            index = np.argmax(preds)
            if index == int(labels[i]):
                rightNumTest += 1

            print(rightNumTest)

            # print(preds)
            # print(index)
            # print(labelDic[int(index)])
            # # print('-'*50)
            # cv2.imshow('', img_)
            # cv2.waitKey()



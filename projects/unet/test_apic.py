from dataload.dataloader import ListDataset
from config.config import load_config, cfg
from net.unet import UNet
from loss.L1L2loss import Regularization
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import cv2

if __name__ == '__main__':
    """config"""
    load_config(cfg, "./config/config.yaml")
    print(cfg)
    device = torch.device('cuda:0')

    """dataset"""
    trainData = ListDataset(trainAnnoPath =cfg.dir.trainAnnoDir,  trainImgPath = cfg.dir.trainImgDir,
                            netInputSizehw = cfg.model.netInput,  augFlag=cfg.data.augment,
                            imgChannelNumber = cfg.data.imgChannelNum, maskChannelNumber = cfg.data.maskChannelNum,
                            normalize = cfg.data.normalize)
    trainLoader = torch.utils.data.DataLoader(
        trainData,
        batch_size=1,#cfg.train.batchSize,
        shuffle=True,
        num_workers=cfg.train.workers,
        pin_memory=True,  # 如果机器计算能力好的话，就可以设置为True，
    )


    """准备网络"""
    network = UNet(cfg.data.imgChannelNum, cfg.model.clsNum, bilinear=True)
    network.to(device)
    if cfg.dir.modelReloadFlag:
        weights = torch.load(cfg.dir.modelSaveDir + cfg.dir.modelName)  # 加载参数
        network.load_state_dict(weights)  # 给自己的模型加载参数

    for e in range(1, 1+ cfg.train.epoch):
        for id, infos in enumerate(trainLoader):
            """dataX"""
            imgs = infos['images'].to(device).float()
            mean = torch.tensor(cfg.data.normalize[0]).cuda().reshape(3, 1, 1)
            std = torch.tensor(cfg.data.normalize[1]).cuda().reshape(3, 1, 1)
            imgs = (imgs - mean) / std

            imgs_ = infos['images']
            image = imgs_[0]
            image = image.to('cpu').numpy()
            image = image.transpose(1, 2, 0).astype(np.uint8)
            image = cv2.UMat(image).get()
            """dataY"""
            masks  = infos['masks'].to(device).type(torch.long)

            """pred"""
            pred = network(imgs)
            pred = torch.softmax(pred, dim = 1)
            with torch.no_grad():
                pred = pred[0]
                bg = pred[0]
                bg = bg.to("cpu").numpy()
                bg = bg < 0.7
                bg = bg.astype(np.uint8)
                bg = bg * 255
                cv2.imshow("pred", bg)
                image[image>0] = 255
                cv2.imshow("orign", image)
                cv2.waitKey()
                print("done")
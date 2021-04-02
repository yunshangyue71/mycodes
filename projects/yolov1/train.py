from data.dataloader_detection import ListDataset
from data.collate_function import collate_function
from config.config import load_config, cfg
from net.resnet import ResNet, ResnetBasic,ResnetBasicSlim
from loss.yololoss import yoloLoss
from dataY.yolov1_dataY import DataY
from loss.L1L2loss import Regularization

from torch.utils.data import SubsetRandomSampler
import numpy as np

import torch
from torch import optim
from torch import nn
import math

if __name__ == '__main__':
    """config"""
    load_config(cfg, "./config/config.yaml")
    print(cfg)
    device = torch.device('cuda:0')

    """dataset"""
    trainData = ListDataset(trainAnnoPath =cfg.dir.trainAnnoDir,  trainImgPath = cfg.dir.trainImgDir,
                            netInputSizehw = cfg.model.netInput,  augFlag=cfg.data.augment,
                            normalize = cfg.data.normalize, imgChannelNumber=cfg.model.imgChannelNumber)
    trainLoader = torch.utils.data.DataLoader(
        trainData,
        collate_fn=collate_function,
        batch_size=cfg.train.batchSize,
        shuffle=True,
        num_workers=cfg.train.workers,
        pin_memory=True,  # 如果机器计算能力好的话，就可以设置为True，

    )

    valData = ListDataset(trainAnnoPath =cfg.dir.valAnnoDir,  trainImgPath = cfg.dir.valImgDir,
                            netInputSizehw = cfg.model.netInput,  augFlag=cfg.data.augment,
                            normalize = cfg.data.normalize, imgChannelNumber=cfg.model.imgChannelNumber)
    valLoader = torch.utils.data.DataLoader(
        trainData,
        collate_fn=collate_function,
        batch_size=cfg.train.batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,  # 如果机器计算能力好的话，就可以设置为True，
    )
    datay = DataY(inputHW = cfg.model.netInput,  # 指定了inputsize 这是因为输入的是经过resize后的图片
                  gride = cfg.model.featSize, # 将网络输入成了多少个网格
                  stride = cfg.model.stride,
                  boxNum = cfg.model.bboxPredNum,
                  clsNum = cfg.model.clsNum)

    """准备网络"""
    network = ResNet(ResnetBasicSlim,
                     #[2, 2, 2, 2],
                     [3,4,6,3],
                     channel_in=cfg.data.imgChannelNumber,
                     channel_out=(cfg.model.bboxPredNum * 5 + cfg.model.clsNum))
    network.to(device)
    if cfg.dir.modelReloadFlag:
        weights = torch.load(cfg.dir.modelSaveDir + cfg.dir.modelName)  # 加载参数
        network.load_state_dict(weights)  # 给自己的模型加载参数

    """指定loss"""
    lossF = yoloLoss(boxNum = cfg.model.bboxPredNum,
                 clsNum = cfg.model.clsNum,
                     lsNoObj=cfg.loss.noobj,
                     lsConf=cfg.loss.conf,
                     lsObj=cfg.loss.obj,
                     lsCls=cfg.loss.cls,
                     lsBox =  cfg.loss.box
                     )

    """其余"""
    optimizer = torch.optim.Adam(network.parameters(), lr=cfg.train.lr0)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=cfg.train.lrPatience)
    warmUpFlag = True if cfg.train.warmupBatch is not None else False
    warmUpIter = 0

    for e in range(1, 1+ cfg.train.epoch):
        """set lr"""
        if not warmUpFlag:
            lr = (cfg.train.lr0 * (pow(cfg.train.lrReduceFactor, (e) // cfg.train.lrReduceEpoch)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


        trainvalLoader = {"train": trainLoader,  "val": valLoader}
        for key, value in trainvalLoader.items():
            for id, infos in enumerate(value):
                if key == "val" and id >=0:
                    break

                """warmup"""
                if warmUpFlag:
                    if warmUpIter < cfg.train.warmupBatch:
                        lr = cfg.train.warmupLr0 + cfg.train.lr0 * (warmUpIter) / cfg.train.warmupBatch
                    elif warmUpIter == cfg.train.warmupBatch:
                        lr = cfg.train.lr0
                        warmUpFlag = False
                    # else:
                        # lr = cfg.train.lr0
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    warmUpIter += 1

                """dataX"""
                imgs = infos['images'].to(device).float()
                mean = torch.tensor(cfg.data.normalize[0]).cuda().reshape(3, 1, 1)
                std = torch.tensor(cfg.data.normalize[1]).cuda().reshape(3, 1, 1)
                imgs = (imgs - mean) / std


                """pred"""
                pred = network(imgs)

                """dataY"""
                bboxesGt = infos['bboxesGt']
                classesGt = infos['classes']
                target = datay.do2(bboxesGt, classesGt, pred)

                """cal loss"""
                lsInfo = lossF.do(pred, target)
                loss = lsInfo["conf"] * cfg.loss.conf+  lsInfo["box"]  * cfg.loss.box+  lsInfo["cls"] * cfg.loss.cls * bool(cfg.model.clsNum-1)
                loss = loss/cfg.train.batchSize
                l1, l2 = Regularization(network)
                loss += cfg.loss.l2 * l2

                if key == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(network.parameters(), 0.5) # gradient clip
                    optimizer.step()
                    #scheduler.step(loss) #可以使其他的指标

                with torch.no_grad():
                    lossS = torch.clone(loss).to('cpu').numpy()
                    lsConf = torch.clone(lsInfo['conf']).to('cpu').numpy()
                    lsBox = torch.clone(lsInfo['box']).to('cpu').numpy()
                    lsCls = torch.clone(lsInfo['cls']).to('cpu').numpy()
                    if id%30==0:
                        print(key, id,"-",int(len(trainData)/cfg.train.batchSize),"/",e,
                          " loss:",lossS, " lsConf:",lsConf, " lsCls:",lsCls, " lsBox:", lsBox,
                          " lr:", lr)


        if e % 1 == 0:
            """参数"""
            savePath = cfg.dir.modelSaveDir + str(e) + '.pth'
            torch.save(network.state_dict(), savePath)  # save
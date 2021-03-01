from dataload.dataloader import ListDataset
from config.config import load_config, cfg
from net.unet import UNet
from loss.L1L2loss import Regularization

import torch
import torch.nn as nn
from torch import optim

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
        batch_size=cfg.train.batchSize,
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

    """指定loss"""
    if cfg.model.clsNum> 1:
        lossF = nn.CrossEntropyLoss()
    else:
        lossF = nn.BCEWithLogitsLoss()

    """其余"""
    optimizer = torch.optim.Adam(network.parameters(), lr=cfg.train.lr0)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if cfg.model.clsNum > 1 else 'max',
    #                                                  factor=cfg.train.lrReduceFactor, patience=cfg.train.lrPatience)
    warmUpFlag = True if cfg.train.warmupBatch is not None else False
    warmUpIter = 0

    for e in range(1, 1+ cfg.train.epoch):
        """set lr"""
        if not warmUpFlag:
            lr = (cfg.train.lr0 * (pow(cfg.train.lrReduceFactor, (e) // cfg.train.lrReduceEpoch)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for id, infos in enumerate(trainLoader):
            """warmup"""
            if warmUpFlag:
                warmUpIter += 1
                if warmUpIter <= cfg.train.warmupBatch:
                    lr = cfg.train.warmupLr0 + cfg.train.lr0 * (warmUpIter-1) / cfg.train.warmupBatch
                else:
                    warmUpFlag = False
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            """dataX"""
            imgs = infos['images'].to(device).float()
            mean = torch.tensor(cfg.data.normalize[0]).cuda().reshape(3, 1, 1)
            std = torch.tensor(cfg.data.normalize[1]).cuda().reshape(3, 1, 1)
            imgs = (imgs - mean) / std

            """dataY"""
            masks  = infos['masks'].to(device).type(torch.long)

            """pred"""
            pred = network(imgs)

            """post"""
            pred = pred.permute(0, 2, 3, 1)
            """cal loss"""
            loss = lossF(pred.reshape(-1, cfg.model.clsNum), masks.reshape(-1))
            l1, l2 =  Regularization(network)
            loss += cfg.loss.l2 * l2

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(network.parameters(), 0.3) # gradient clip
            optimizer.step()
            #scheduler.step(loss)  # 可以使其他的指标

            with torch.no_grad():
                if 1:
                    lossS = torch.clone(loss).to('cpu').numpy()
                    print(id,"/",e, " loss:", lossS, " lr:", lr)
                if e % 1 == 0:
                    """参数"""
                    savePath = cfg.dir.modelSaveDir + str(e) + '.pth'
                    torch.save(network.state_dict(), savePath)  # save
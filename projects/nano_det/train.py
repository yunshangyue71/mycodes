import torch
import cv2
import numpy as np

from data.dataloader_detection import ListDataset
from net.net import NanoNet
from data.collate_function import collate_function
from anchor.anchorbox_generate import AnchorGenerator
from anchor.predBox_and_imageBox import distance2bbox, bbox2distance
from anchor.anchor_assigner import Assigner
from utils.bbox_distribution import BoxesDistribution
from loss.giou_loss import GIoULoss
from loss.distribution_focal_loss import DistributionFocalLoss
from loss.quality_focal_loss import QualityFocalLoss
from loss.multi_iou_cal import MultiIoUCal
from config.config import load_config, cfg
from loss.L1L2loss import Regularization
from torch import optim
import torch.nn as nn


if __name__ == '__main__':
    """config"""
    load_config(cfg, "./config/config.yaml")
    print(cfg)
    headerNum = len(cfg.model.featSizes)
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
    """END"""

    """准备网络"""
    network = NanoNet(classNum=cfg.model.classNum,
                      imgChannelNum = cfg.data.imgChannelNumer,
                      regBoxNum = cfg.model.bboxPredNum)
    network.to(device)
    if cfg.dir.modelReloadPath is not None:
        weights = torch.load(cfg.dir.modelReloadPath)#加载参数
        network.load_state_dict(weights)#给自己的模型加载参数
    optimizer = torch.optim.Adam(network.parameters(), lr=cfg.train.lr0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if cfg.model.clsNum > 1 else 'max', patience=2)

    """准备一张照片level 的anchorBoxes"""
    anchorBoxes = []
    for i in range(headerNum):
        gen = AnchorGenerator(wsizes=[5*cfg.model.strides[i]], hsizes=[5*cfg.model.strides[i]] )
        anchorBoxes.append(gen.genAnchorBoxes(featmapSizehw=cfg.model.featSizes[i], stride = cfg.model.strides[i]))

    warmUpFlag = True if cfg.train.warmupBatch is not None else False
    warmUpIter = 0
    for e in range(1, 1+ cfg.train.epoch):
        """set lr"""
        if not warmUpFlag:
            lr = (cfg.train.lr0 / (pow(3, (e) // 3)))
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

            """forward and pred"""
            imgs = infos['images'].to(device).float()
            mean = torch.tensor(cfg.data.normalize[0]).cuda().reshape(3, 1, 1)
            std = torch.tensor(cfg.data.normalize[1]).cuda().reshape(3, 1, 1)
            imgs = (imgs - mean) / std

            bboxesGt = infos['bboxesGt']
            classesGt = infos['classes']
            boxPred, clsPred = network(imgs)
            realBatchSize = imgs.shape[0]  # 最后一个batch可能数目不够

            """anchor labeling1,每一个level labeling 一次每一个anchor应该回归出来的东西"""
            boxGtAnchor = []
            clsGtAnchor = []
            for imgId in range(imgs.shape[0]):
                bboxesGtImg = torch.from_numpy(bboxesGt[imgId]).to(device)
                bboxesGtImg[:, 2:] += bboxesGtImg[:, :2]
                classesGtImg = torch.from_numpy(classesGt[imgId]).to(device).view(-1)

                anchorBoxes_ = [i.reshape(-1, 4) for i in anchorBoxes]
                anchorBoxes_ = torch.cat(anchorBoxes_, dim=0)
                assign = Assigner(9, anchorBoxes_, bboxesGtImg, classesGtImg)
                assignInfos = assign.master()

                boxGtAnchor.append(assignInfos['anchorBoxGt'])
                clsGtAnchor.append(assignInfos['anchorBoxGtClass'])
            boxGtAnchor_ = torch.stack(boxGtAnchor, 0)
            clsGtAnchor_ = torch.stack(clsGtAnchor, 0)

            boxGtAnchor = []
            clsGtAnchor = []
            start = 0
            end = cfg.model.featSizes[0][0] * cfg.model.featSizes[0][1]
            for level in range(headerNum):
                boxGtAnchor.append(boxGtAnchor_[:,start:end,:])
                clsGtAnchor.append(clsGtAnchor_[:,start:end])
                if level == 2:
                    break
                start = end
                end += cfg.model.featSizes[level+1][0]*cfg.model.featSizes[level+1][1]

            """to show anchor assigned results"""
            showFlag = 0
            """if showFlag:
                for i in range(realBatchSize):
                    image = torch.clone(imgs[i])
                    image = image.to('cpu').numpy()
                    image = image.transpose(1, 2, 0)#*255
                    image = image.astype(np.uint8)
                    image = cv2.UMat(image).get()
                    for levelId in range(headerNum):
                        imageID = np.copy(image).astype(np.uint8)
                        cls = anchorBoxGtClass[levelId]
                        box = anchorBoxGt[levelId]
                        cls = cls.to('cpu').numpy()
                        box = box.to('cpu').numpy()
                        clsi = cls[i]
                        boxi = box[i]
                        for j in range(clsi.shape[0]):
                            if clsi[j] != -1:
                                feath, featw = cfg.model.featSizes[levelId]
                                print(boxi[j], clsi[j], j//featw, j%featw,j)
                                cv2.rectangle(imageID, (int(boxi[j][0]),int(boxi[j][1])),
                                              (int(boxi[j][2]),int(boxi[j][3])), (0, 0, 255), 1)
                                cv2.circle(imageID, (int(j%featw)*cfg.model.strides[levelId] + int(cfg.model.strides[levelId]/2),
                                                   int(j//featw)*cfg.model.strides[levelId] + int(cfg.model.strides[levelId]/2)),
                                           2, (0, 0, 255), -1)
                                cv2.putText(imageID, str(clsi[j]), (int(boxi[j][0]),int(boxi[j][1])), 1, 1, (0,255,0))
                        cv2.imshow('img'+str(levelId), imageID)
                        cv2.waitKey()
                print("show anghor box gt")
"""
            loss = []
            for level in range(headerNum):
                feath, featw = cfg.model.featSizes[level]
                pointsx = torch.arange(0, featw, device=device).repeat(feath).reshape(feath, featw).to(device)
                pointsy = torch.arange(0, feath, device=device).repeat(featw).reshape(featw, feath).to(device)
                pointsy = pointsy.t()
                points = torch.stack((pointsx, pointsy), dim=2).reshape(-1, 2).repeat(realBatchSize, 1, 1). \
                             reshape(-1, 2) * cfg.model.strides[level] + cfg.model.strides[level] / 2
                # 所有的形状（B * W * H , ...）， 标准都是像素， 非gride
                """target"""
                clsGtAnchorLevel = clsGtAnchor[level].reshape(-1)  # ->shape(b * feath * featw)

                boxGtAnchorLevel = boxGtAnchor[level].reshape(-1, 4)         # target 像素坐标系 ->shape(bachsize * feath * featw, 4)
                boxGtAnchorDist = bbox2distance(points, boxGtAnchorLevel)    # target 距离， 像素单位

                """pred"""
                clsPredLevel = clsPred[level].reshape(-1, cfg.model.classNum)
                boxPredLevel = boxPred[level].reshape(-1, 4 * cfg.model.bboxPredNum)             # pred 概率

                # box 预测的是 到 anchor点的gride距离是0，1，2，3，4，5，6，7的概率
                boxPredLevelOne = BoxesDistribution(reg_max=cfg.model.bboxPredNum)(boxPredLevel)    # pred 距离, gride
                boxPredLevelOne *= cfg.model.strides[level]                                         # pred 距离， 像素
                boxPredLevelLUOne = distance2bbox(points, boxPredLevelOne)                          # pred 像素坐标系

                """pred pos """
                posindex = torch.nonzero(clsGtAnchorLevel + 1).reshape(-1) # background 的标签是-1，

                posBoxPredLevel = boxPredLevel[posindex]            # 概率
                posBoxPredLevelOne = boxPredLevelOne[posindex]      # 距离
                posBoxPredLevelLUOne = boxPredLevelLUOne[posindex]  # 像素坐标系
                posClsPredLevel = clsPredLevel[posindex]

                """target pos"""
                posClsGtAnchorLevel = clsGtAnchorLevel[posindex]    #

                posBoxGtAnchorLevel = boxGtAnchorLevel[posindex]    # 像素坐标系
                posBoxGtAnchorDist = boxGtAnchorDist[posindex]      # 距离

                if len(posindex) > 0:
                    weight = posClsPredLevel.sigmoid().max(dim=1)[0]
                    weight4 = torch.stack([weight, weight, weight, weight], dim = 1).reshape(-1)

                    """giou loss (posNum, )"""
                    lossGiouLevel = GIoULoss()(posBoxGtAnchorLevel, posBoxPredLevelLUOne)
                    lossGiouLevel_ = lossGiouLevel * weight
                    lossGiouLevel = lossGiouLevel.sum()

                    """dflLoss (batch posNum * 4, )"""
                    a = posBoxPredLevel.reshape(-1, cfg.model.bboxPredNum)
                    b = posBoxGtAnchorDist.reshape(-1) / cfg.model.strides[level]
                    b = b.clamp(0, cfg.model.bboxPredNum - 1 - 1e-5)
                    lossDfLevel = DistributionFocalLoss(reduction='none').cal(a, b)  # 这里一共8个位置， 每个点距离anchor点可能是0-7，共8个距离，
                    lossDfLevel *= weight4
                    lossDfLevel = lossDfLevel.sum()

                else:
                    lossGiouLevel = boxPredLevel.sum() * 0
                    lossDfLevel = boxPredLevel.sum() * 0
                    weight = torch.tensor(0).cuda()
                    weight4 = torch.tensor(0).cuda()

                """qfLoss """
                posQuality = MultiIoUCal(posBoxPredLevelLUOne, posBoxGtAnchorLevel,
                                       mode='giou', isAligned=True).iouResult().clamp(min=1e-6)
                quality = torch.zeros(size=(realBatchSize * featw * feath,), device=device)
                quality[posindex] = posQuality

                lossQfLevel = QualityFocalLoss().cal(clsPredLevel, (clsGtAnchorLevel, quality))

                batchPosNum = torch.nonzero((torch.cat(clsGtAnchor, dim = 1) + 1).reshape(-1)).numel()
                lossQfLevel = lossQfLevel.sum() / batchPosNum

                loss.append([lossGiouLevel, weight.sum()])
                loss.append([lossDfLevel, weight4.sum()])
                loss.append([lossQfLevel, batchPosNum])

            l1, l2 = Regularization(network)

            giouloss = (loss[0][0] + loss[3][0] + loss[6][0]) / (loss[0][1] + loss[3][1] + loss[6][1])
            dfloss =   (loss[1][0] + loss[4][0] + loss[7][0]) / (loss[1][1] + loss[4][1] + loss[7][1])
            qfloss =   (loss[2][0] + loss[5][0] + loss[8][0]) # / (loss[2][1] )#+ loss[5][1] + loss[8][1])

            loss_ = cfg.train.giouloss * giouloss + \
                    cfg.train.dfloss * dfloss + \
                    cfg.train.qfloss * qfloss + \
                    cfg.train.l2 * l2
            optimizer.zero_grad()
            loss_.backward()
            nn.utils.clip_grad_value_(network.parameters(), 0.1) # gradient clip
            optimizer.step()
            scheduler.step(loss_)  # 可以使其他的指标

            #1个epoch print
            with torch.no_grad():
                lossp = torch.clone(loss_)
                gioulossp = torch.clone(giouloss)
                dflossp = torch.clone(dfloss)
                qflossp = torch.clone(qfloss)
                print(e, "total:", lossp.to('cpu').numpy(),
                        " giouloss:", gioulossp.to('cpu').numpy(),
                        " dfloss:",dflossp.to('cpu').numpy(),
                        " qfloss:",qflossp .to('cpu').numpy(),
                      " lr: ", lr,loss[4][1], loss[5][1])

        if e % 5 == 0:
            """参数"""
            savePath =cfg.dir.modelSaveDir+str(e)+'.pth'
            torch.save(network.state_dict(), savePath)#save
import torch
import cv2
import numpy as np

from data.dataloader_detection_hand import ListDataset
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
"""config"""
load_config(cfg, "./config/config_hand.yaml")
print(cfg)
headerNum = len(cfg.model.featSizes)
device = torch.device('cuda:0')

"""dataset"""
trainData = ListDataset(trainAnnoPath =cfg.dir.trainAnnoDir,  trainImgPath = cfg.dir.trainImgDir,
                        netInputSizehw = cfg.model.netInput,  augFlag=cfg.augment.flag)
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
network = NanoNet(classNum=cfg.model.classNum)
network.to(device)
if cfg.dir.modelReloadPath is not None:
    weights = torch.load(cfg.dir.modelReloadPath)#加载参数
    network.load_state_dict(weights)#给自己的模型加载参数
optimizer = torch.optim.Adam(network.parameters(), lr=cfg.train.lr0)

"""准备一张照片level 的anchorBoxes"""
anchorBoxes = []
for i in range(headerNum):
    gen = AnchorGenerator(wsizes=[5*cfg.model.strides[i]], hsizes=[5*cfg.model.strides[i]] )
    anchorBoxes.append(gen.genAnchorBoxes(featmapSizehw=cfg.model.featSizes[i], stride = cfg.model.strides[i]))


for e in range(1, 1+ cfg.train.epoch):
    """warmup"""
    if e <= cfg.train.warmupEpoch:
        lr = cfg.train.warmupLr0 + cfg.train.lr0 * (e-1) / cfg.train.warmupEpoch
    else:
        lr = (cfg.train.lr0 / (pow(3, (e - cfg.train.warmupEpoch) // 7)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    """END"""

    for id, infos in enumerate(trainLoader):
        """forward and pred"""
        imgs = infos['images']
        bboxesGt = infos['bboxesGt']
        classesGt = infos['classes']
        imgs = imgs.to(device).float()
        bboxesPred, classesPred = network(imgs)
        realBatchSize = imgs.shape[0]  # 最后一个batch可能数目不够

        """anchor labeling1,每一个level labeling 一次每一个anchor应该回归出来的东西"""
        anchorBoxGt = []
        anchorBoxGtClass = []
        for imgId in range(imgs.shape[0]):
            bboxesGtImg = torch.from_numpy(bboxesGt[imgId]).to(device)
            bboxesGtImg[:, 2:] += bboxesGtImg[:, :2]
            classesGtImg = torch.from_numpy(classesGt[imgId]).to(device).view(-1)

            anchorBoxes_ = [i.reshape(-1, 4) for i in anchorBoxes]
            anchorBoxes_ = torch.cat(anchorBoxes_, dim=0)
            assign = Assigner(3, anchorBoxes_, bboxesGtImg, classesGtImg)
            assignInfos = assign.master()

            anchorBoxGt.append(assignInfos['anchorBoxGt'])
            anchorBoxGtClass.append(assignInfos['anchorBoxGtClass'])
        anchorBoxGt_ = torch.stack(anchorBoxGt, 0)
        anchorBoxGtClass_ = torch.stack(anchorBoxGtClass, 0)
        anchorBoxGt = []
        anchorBoxGtClass = []
        start = 0
        end = cfg.model.featSizes[0][0]*cfg.model.featSizes[0][1]
        for level in range(headerNum):
            anchorBoxGt.append(anchorBoxGt_[:,start:end,:])
            anchorBoxGtClass.append(anchorBoxGtClass_[:,start:end])
            if level == 2:
                break
            start = end
            end += cfg.model.featSizes[level+1][0]*cfg.model.featSizes[level+1][1]

        """to show anchor assigned results"""
        showFlag = 0
        if showFlag:
            for i in range(realBatchSize):
                image = torch.clone(imgs[i])
                image = image.to('cpu').numpy()
                image = image.transpose(1, 2, 0)*255
                image = image.astype(np.uint8)
                image = cv2.UMat(image).get()
                for levelId in range(headerNum):
                    imageID = np.copy(image)
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

        loss = []
        for level in range(headerNum):
            feath, featw = cfg.model.featSizes[level]
            anchorBoxGtLevel = anchorBoxGt[level].reshape(-1, 4) #->shape(bachsize * feath * featw, 4)
            anchorBoxGtClassLevel = anchorBoxGtClass[level].reshape(-1) #->shape(b * feath * featw)
            bboxesPredLevel = bboxesPred[level].reshape(-1, 4 * cfg.model.bboxPredNum) #->shape()
            classesPredLevel = classesPred[level]

            posindex = torch.nonzero(anchorBoxGtClassLevel + 1).reshape(-1)
            # posindex = torch.nonzero(torch.zeros_like(anchorBoxGtClassLevel )).reshape(-1)# 检测当没有pos时候是否有bug
            '''giou loss (posNum, )'''
            #box 预测的是 到 anchor点的gride距离是0，1，2，3，4，5，6，7的概率
            bboxesPredLevelOne = BoxesDistribution(reg_max=cfg.model.bboxPredNum)(bboxesPredLevel)*cfg.model.strides[level]
            pointsx = torch.arange(0, featw, device=device).repeat(feath).reshape(feath, featw).to(device)
            pointsy = torch.arange(0, feath, device=device).repeat(featw).reshape(featw, feath).to(device)
            pointsy = pointsy.t()
            points = torch.stack((pointsx, pointsy), dim = 2).reshape(-1, 2).repeat(realBatchSize, 1, 1).\
                         reshape(-1, 2)*cfg.model.strides[level] + cfg.model.strides[level]/2

            bboxesPredLevelLUDistOne = distance2bbox(points, bboxesPredLevelOne)
            #这里还是相对与anchor 点的距离，还没转换成响度与左上角的距离
            #bboxesPredLUDistLevelOne = LUDist2AnchorPointDist (bboxesPredLUDistLevelOne, posindex)
            giouLossLevel = GIoULoss()(anchorBoxGtLevel[posindex], bboxesPredLevelLUDistOne[posindex])

            """dflLoss (batch posNum * 4, )"""
            bboxesPredLevel = bboxesPred[level].reshape(-1, 4* cfg.model.bboxPredNum)
            a = bboxesPredLevel[posindex].reshape(-1, cfg.model.bboxPredNum)
            anchorBoxGtLevel = anchorBoxGt[level].reshape(realBatchSize* featw*feath,-1)
            anchorBoxGtDist = bbox2distance(points, anchorBoxGtLevel)#LUDist2AnchorPointDist(anchorBoxGtLevel, featSizes[level], strides[level])
            b = anchorBoxGtDist.reshape(-1, 4)[posindex].reshape(-1) / (cfg.model.strides[level])#*5./(pow(2,level)))# 这里面有时候有负的????
            dfLossLevel = DistributionFocalLoss(reduction = 'none').\
                cal(a, b.clamp(0, cfg.model.bboxPredNum-1-1e-5))#.clamp(0, bboxPredNum-1-1e-5)#如果设置为-8， 由于里面的是float32的那么就会，四舍五入为0
            #a keyi shi fu shu ?
            # 这里一共8个位置， 每个点距离anchor点可能是0-7，共8个距离，

            """qfLoss (batch*featw*feath, )"""
            #class 就是看定位是不是定位的准
            classesPredLevel = classesPredLevel.reshape(-1, cfg.model.classNum)
            # anchorBoxGtClass
            quality_ = MultiIoUCal(bboxesPredLevelLUDistOne[posindex], anchorBoxGtLevel.reshape(-1, 4)[posindex],
                                   mode='giou', isAligned=True).iouResult().clamp(min=1e-6)

            quality = torch.zeros(size=(realBatchSize*featw*feath,), device=device)
            quality[posindex] = quality_
            qfLossLevel = QualityFocalLoss().cal(classesPredLevel, (anchorBoxGtClass[level].reshape(-1), quality))
            #print('------------giou:', giouLossLevel.mean(),'dloss:',dfLossLevel.mean(), 'qloss:', qfLossLevel.mean())
            loss.append([giouLossLevel.sum(),len(posindex)])
            loss.append([dfLossLevel.sum(), len(posindex)])
            loss.append([qfLossLevel.sum(), feath*featw])
        l1, l2 = Regularization(network)
        optimizer.zero_grad()
        giouloss = (loss[0][0] + loss[3][0] + loss[6][0]) / (loss[0][1] + loss[3][1] + loss[6][1])
        dfloss =   (loss[1][0] + loss[4][0] + loss[7][0]) / (loss[1][1] + loss[4][1] + loss[7][1])
        qfloss =   (loss[2][0] + loss[5][0] + loss[8][0]) / (loss[2][1] + loss[5][1] + loss[8][1])

        loss_ = cfg.train.giouloss * giouloss + \
                cfg.train.dfloss * dfloss + \
                cfg.train.qfloss * qfloss + \
                cfg.train.l2 * l2
        loss_.backward()
        optimizer.step()

    #1个epoch print
    with torch.no_grad():
        lossp = torch.clone(loss_)
        gioulossp = torch.clone(giouloss)
        dflossp = torch.clone(dfloss)
        qflossp = torch.clone(qfloss)
        print(e, "total:", lossp.to('cpu').numpy(),
                " giouloss:", gioulossp.to('cpu').numpy(),
                " dfloss:",dflossp.to('cpu').numpy(),
                " qfloss:",qflossp .to('cpu').numpy())

    if e % 5 == 0:
        """参数"""
        savePath =cfg.dir.modelSaveDir+str(e)+'.pth'
        torch.save(network.state_dict(), savePath)#save



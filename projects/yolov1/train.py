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
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import time

if __name__ == '__main__':
    """config"""
    cfgpath = "./config/config.yaml"
    load_config(cfg, cfgpath)
    print(cfg)
    device = torch.device('cuda:0')

    """dataset"""
    trainData = ListDataset(trainAnnoPath=cfg.dir.trainAnnoDir, trainImgPath=cfg.dir.trainImgDir,
                            netInputSizehw=cfg.model.netInput, augFlag=cfg.data.augment,
                            imgChannelNumber=cfg.model.imgChannelNumber, clsname=cfg.clsname)
    trainLoader = torch.utils.data.DataLoader(
        trainData,
        collate_fn=collate_function,
        batch_size=cfg.train.batchSize,
        shuffle=True,
        num_workers=cfg.train.workers,
        pin_memory=True,  # 如果机器计算能力好的话，就可以设置为True，

    )

    valData = ListDataset(trainAnnoPath=cfg.dir.valAnnoDir, trainImgPath=cfg.dir.valImgDir,
                          netInputSizehw=cfg.model.netInput, augFlag=cfg.data.augment,
                          imgChannelNumber=cfg.model.imgChannelNumber)
    valLoader = torch.utils.data.DataLoader(
        trainData,
        collate_fn=collate_function,
        batch_size=cfg.train.batchSize,
        shuffle=cfg.data.shuffle,
        num_workers=0,
        pin_memory=True,  # 如果机器计算能力好的话，就可以设置为True，
    )
    datay = DataY(inputHW=cfg.model.netInput,  # 指定了inputsize 这是因为输入的是经过resize后的图片
                  gride=cfg.model.featSize,  # 将网络输入成了多少个网格
                  stride=cfg.model.stride,
                  boxNum=cfg.model.bboxPredNum,
                  clsNum=cfg.model.clsNum)

    """准备网络"""
    network = ResNet(ResnetBasicSlim,
                     # [2, 2, 2, 2],
                     [3, 4, 6, 3],
                     channel_in=cfg.data.imgChannelNumber,
                     channel_out=(cfg.model.bboxPredNum * 5 + cfg.model.clsNum))
    # network = YOLOv1(params={"dropout": 0.5, "num_class": cfg.model.clsNum})
    network.to(device)
    startEpoch = 1
    if cfg.dir.modelReloadFlag:
        savedDict = torch.load(cfg.dir.modelSaveDir + cfg.dir.modelName)  # 加载参数
        weights =  savedDict['savedModel']
        startEpoch = savedDict['epoch']+1
        network.load_state_dict(weights)  # 给自己的模型加载参数

    """指定loss"""
    lossF = yoloLoss(boxNum=cfg.model.bboxPredNum,
                     clsNum=cfg.model.clsNum,
                     lsNoObj=cfg.loss.noobj,
                     lsConf=cfg.loss.conf,
                     lsObj=cfg.loss.obj,
                     lsCls=cfg.loss.cls,
                     lsBox=cfg.loss.box
                     )

    """optimizer"""
    optimizer = torch.optim.Adam(network.parameters(), lr=cfg.train.lr0)
    # optimizer = torch.optim.SGD(network.parameters(), lr=cfg.train.lr0)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=cfg.train.lrPatience)

    """lr warm up"""
    if cfg.train.warmupBatch is not None and not cfg.dir.modelReloadFlag:# reload  not warmup
        warmUpFlag = True
    else:
        warmUpFlag = False
    warmUpIter = 0
    warmUpBatch = min(cfg.train.warmupBatch,len(trainLoader))# set the warmup batch num up limit

    """log """
    logtime = time.asctime(time.localtime(time.time()))
    logdir = cfg.dir.logSaveDir + "start_epoch_" + str(startEpoch) + " time_"+str(logtime) +  "/"
    writer = SummaryWriter(logdir)
    addGraphFlag = True
    addImgFlag = True
    addCofigFlag = True
    """打开 
    tensorboard - -logdir
    runs
    打开
    localhost: 6006"""

    """start to train """
    batchNum = len(trainLoader) * cfg.train.epoch
    for e  in range(startEpoch, cfg.train.epoch):
        """set decay lr"""
        if not warmUpFlag:
            lr = (cfg.train.lr0 * (pow(cfg.train.lrReduceFactor, (e) // cfg.train.lrReduceEpoch)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for id, infos in enumerate(trainLoader):
            """warmup"""
            if warmUpFlag:
                if warmUpIter < warmUpBatch:
                    lr = cfg.train.warmupLr0 + cfg.train.lr0 * (warmUpIter) / warmUpBatch
                elif warmUpIter == warmUpBatch:
                    lr = cfg.train.lr0
                    warmUpFlag = False
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                warmUpIter += 1

            """dataX"""
            images = infos['images'].to(device).float()
            mean = torch.tensor(cfg.data.normalize[0]).cuda().reshape(3, 1, 1)
            std = torch.tensor(cfg.data.normalize[1]).cuda().reshape(3, 1, 1)
            imgs = (images - mean) / std

            """pred"""
            pred = network(imgs)

            """dataY"""
            bboxesGt = infos['bboxesGt']
            classesGt = infos['classes']
            target = datay.do2(bboxesGt, classesGt, pred)
            # if classesGt[0][0][0] == -1:
            #     print("")

            """cal loss"""
            lsInfo = lossF.do(pred, target)
            loss = lsInfo["conf"] * cfg.loss.conf + lsInfo["box"] * cfg.loss.box + lsInfo[
                "cls"] * cfg.loss.cls * bool(cfg.model.clsNum - 1)
            loss = loss / cfg.train.batchSize
            l1, l2 = Regularization(network)
            loss += cfg.loss.l2 * l2

            """backward"""
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(network.parameters(), 0.5)  # gradient clip
            optimizer.step()
            # scheduler.step(loss) #可以使其他的指标

            """print"""
            with torch.no_grad():
                lossS = torch.clone(loss).to('cpu').numpy()
                lsConf = torch.clone(lsInfo['conf']).to('cpu').numpy()
                lsBox = torch.clone(lsInfo['box']).to('cpu').numpy()
                lsCls = torch.clone(lsInfo['cls']).to('cpu').numpy()
                if id % 30 == 0:
                    niter = e*len(trainLoader) + id+1
                    print("[bc:{}/{} e: {}/{} total_bc:{} per:{:.3f}%]".\
                          format(id,len(trainLoader), e,cfg.train.epoch, batchNum,
                                 float(niter*100)/batchNum ),
                          " loss:%.4f" % lossS, " lsConf:%.4f"% lsConf, " lsCls:%.4f"% lsCls, " lsBox:%.4f"% lsBox,"l2:%.4f"%l2,
                          " lr:%.7f"%lr)

            """tensorboardX to view"""
            #loss add per iter
            writer.add_scalars("loss/scalar_group",
                               {"loss":loss,"lossConf":lsInfo['conf'],
                                "lossBox":lsInfo['box'],"lossCls":lsInfo['cls'],
                                "l2Loss":l2, "lr":lr},niter)


            #add graph only once
            if addGraphFlag:
                writer.add_graph(network, imgs)
                addGraphFlag = False

            # add img only once
            if addImgFlag:
                x = vutils.make_grid(images, normalize=True, scale_each=False)
                writer.add_image('Image/origin', x, startEpoch)

                x = vutils.make_grid(imgs, normalize=True, scale_each=False)
                writer.add_image('Image/normalized/', x, startEpoch)

                addImgFlag = False

            #iadd config to text
            if addCofigFlag:
                for key,value in cfg.items():
                    for k, v in value.items():
                        writer.add_text(str(key) + "/" + str(k), str(v))
                addCofigFlag = False

            #add hist per epoch
            if niter % (len(trainLoader)/2) == 0:
                for name, param in network.state_dict().items():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), id)

            #if need , i can add any 4d tensor  to supervise the value


        if e % 1 == 0:
            """参数"""
            savePath = cfg.dir.modelSaveDir + str(e) + 'b.pth'
            saveDict = {"savedModel":network.state_dict(),
                        "epoch":e,
                        }
            # torch.save(network.state_dict(), savePath)  # save
            torch.save(saveDict, savePath)
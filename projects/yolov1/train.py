from data.dataloader_detection import ListDataset
from data.collate_function import collate_function
from config.config import load_config, cfg
from net.resnet import ResNet, ResnetBasic,ResnetBasicSlim
from loss.yololoss import yoloLoss
from dataY.yolov1_dataY import DataY

import torch

if __name__ == '__main__':
    """config"""
    load_config(cfg, "./config/config.yaml")
    print(cfg)
    device = torch.device('cuda:0')

    """dataset"""
    trainData = ListDataset(trainAnnoPath =cfg.dir.trainAnnoDir,  trainImgPath = cfg.dir.trainImgDir,
                            netInputSizehw = cfg.model.netInput,  augFlag=cfg.data.augment,
                            normalize = cfg.data.normalize)
    trainLoader = torch.utils.data.DataLoader(
        trainData,
        collate_fn=collate_function,
        batch_size=cfg.train.batchSize,
        shuffle=True,
        num_workers=cfg.train.workers,
        pin_memory=True,  # 如果机器计算能力好的话，就可以设置为True，
    )
    datay = DataY(inputHW = cfg.model.netInput,  # 指定了inputsize 这是因为输入的是经过resize后的图片
                  gride = cfg.model.featSize, # 将网络输入成了多少个网格
                  stride = cfg.model.stride,
                  boxNum = cfg.model.bboxPredNum,
                  clsNum = cfg.model.clsNum)

    """准备网络"""
    # network = ResNet(ResnetBasic, [2, 2, 2, 2], channel_out = 15)
    network = ResNet(ResnetBasicSlim, [2, 2, 2, 2], channel_out=(cfg.model.bboxPredNum * 5 + cfg.model.clsNum))
    network.to(device)
    if cfg.dir.modelReloadFlag:
        weights = torch.load(cfg.dir.modelSaveDir + cfg.dir.modelName)  # 加载参数
        network.load_state_dict(weights)  # 给自己的模型加载参数

    """指定loss"""
    lossF = yoloLoss(boxNum = cfg.model.bboxPredNum,
                 clsNum = cfg.model.clsNum)

    """其余"""
    optimizer = torch.optim.Adam(network.parameters(), lr=cfg.train.lr0)
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

            """dataX"""
            imgs = infos['images']

            """dataY"""
            bboxesGt = infos['bboxesGt']
            classesGt = infos['classes']
            target = datay.do(bboxesGt, classesGt)

            """pred"""
            imgs = imgs.to(device).float()
            pred = network(imgs)
            loss, lsInfo = lossF.do(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if 1:
                    lossS = torch.clone(loss).to('cpu').numpy()
                    lsConf = torch.clone(lsInfo['conf']).to('cpu').numpy()
                    lsBox = torch.clone(lsInfo['box']).to('cpu').numpy()
                    lsCls = torch.clone(lsInfo['cls']).to('cpu').numpy()
                print(id,"/",e,
                      " loss:",lossS, " lsConf:",lsConf, " lsCls:",lsCls, " lsBox:", lsBox,
                      " lr:", lr)
                if e % 5 == 0:
                    """参数"""
                    savePath = cfg.dir.modelSaveDir + str(e) + '.pth'
                    torch.save(network.state_dict(), savePath)  # save
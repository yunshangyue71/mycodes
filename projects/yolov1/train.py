from data.dataloader_detection import ListDataset
from data.collate_function import collate_function
from config.config import load_config, cfg
from net.resnet import ResNet, ResnetBasic,ResnetBasicSlim
from post.target_post import Post
from loss.yololoss import yoloLoss

import torch

if __name__ == '__main__':
    """config"""
    load_config(cfg, "./config/config.yaml")
    print(cfg)
    headerNum = len(cfg.model.featSizes)
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

    """准备网络"""
    # network = ResNet(ResnetBasic, [2, 2, 2, 2], channel_out = 15)
    network = ResNet(ResnetBasicSlim, [2, 2, 2, 2], channel_out=15)
    network.to(device)
    if cfg.dir.modelReloadPath is not None:
        weights = torch.load(cfg.dir.modelReloadPath)  # 加载参数
        network.load_state_dict(weights)  # 给自己的模型加载参数
    optimizer = torch.optim.Adam(network.parameters(), lr=cfg.train.lr0)

    postF = Post(wsizes=[32], hsizes=[32], featuremapSize = (10,10), stride = 32)
    lossF = yoloLoss()

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
            imgs = infos['images']
            imgs = imgs.to(device).float()
            pred = network(imgs)

            """post"""
            bboxesGt = infos['bboxesGt']
            classesGt = infos['classes']
            p = postF.forward(boxGt = bboxesGt, clsGt = classesGt)

            loss = lossF.forward(pred, bboxesGt)
            realBatchSize = imgs.shape[0]  # 最后一个batch可能数目不够
            raise



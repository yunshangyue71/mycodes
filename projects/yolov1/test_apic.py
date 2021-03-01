from data.dataloader_detection import ListDataset
from data.collate_function import collate_function
from config.config import load_config, cfg
from net.resnet import ResNet, ResnetBasic,ResnetBasicSlim
from utils.nms_np_simple import nms
from dataY.yolov1_dataY import DataY

import numpy as np
import cv2
from loss.yololoss import yoloLoss
from dataY.yolov1_dataY import DataY

import torch

if __name__ == '__main__':
    """config"""
    load_config(cfg, "./config/config.yaml")
    print(cfg)
    device = torch.device('cuda:0')

    scoreThresh = 0.9
    iouThresh = 0.1

    """dataset"""
    trainData = ListDataset(trainAnnoPath=cfg.dir.valAnnoDir, trainImgPath=cfg.dir.trainImgDir,
                            netInputSizehw=cfg.model.netInput, augFlag=False,
                            normalize=cfg.data.normalize, imgChannelNumber=cfg.model.imgChannelNumber)
    trainLoader = torch.utils.data.DataLoader(
        trainData,
        collate_fn=collate_function,
        batch_size= 1, #cfg.train.batchSize,
        shuffle= False,
        num_workers=cfg.train.workers,
        pin_memory=True,  # 如果机器计算能力好的话，就可以设置为True，
    )

    datay = DataY(inputHW=cfg.model.netInput,  # 指定了inputsize 这是因为输入的是经过resize后的图片
                  gride=cfg.model.featSize,  # 将网络输入成了多少个网格
                  stride=cfg.model.stride,
                  boxNum=cfg.model.bboxPredNum,
                  clsNum=cfg.model.clsNum)
    """准备网络"""
    # network = ResNet(ResnetBasic, [2, 2, 2, 2], channel_out = 15)
    network = ResNet(ResnetBasicSlim, [2, 2, 2, 2],
                     channel_in=cfg.data.imgChannelNumber,
                     channel_out=(cfg.model.bboxPredNum * 5 + cfg.model.clsNum))
    network.to(device)
    if 1:
        weights = torch.load(cfg.dir.modelSaveDir + cfg.dir.modelName)  # 加载参数
        network.load_state_dict(weights)  # 给自己的模型加载参数

    with torch.no_grad():
        for id, infos in enumerate(trainLoader):
            """forward and pred"""
            imgs_ = infos['images']
            image = imgs_[0]
            image = image.to('cpu').numpy()
            image = image.transpose(1, 2, 0).astype(np.uint8)
            image = cv2.UMat(image).get()

            imgs = torch.clone(imgs_)
            imgs = imgs_.to(device).float()
            mean = torch.tensor(cfg.data.normalize[0]).cuda().reshape(3, 1, 1)
            std = torch.tensor(cfg.data.normalize[1]).cuda().reshape(3, 1, 1)
            imgs = (imgs - mean) / std

            bboxesGt = infos['bboxesGt']
            classesGt = infos['classes']

            """dataY"""
            bboxesGt = infos['bboxesGt']
            classesGt = infos['classes']
            target = datay.do(bboxesGt, classesGt)

            pred = network(imgs)

            pred = pred.permute(0, 2, 3, 1)  # rotate the BCHW to BHWC

            # pred = target
            # target contain object mask
            coobjMask_ = pred[:, :, :, 4] > scoreThresh
            for i in range(1, cfg.model.bboxPredNum):
                coobjMask_ = torch.logical_or(coobjMask_, pred[:, :, :, 4 + i*5] > scoreThresh)

            '''选择每个anchor 点 预测置信度最大的, bbox num = 1 代码也可以'''
            confPred = pred[:, :, :, 4:5 * cfg.model.bboxPredNum:5]
            maxNum, maxIndex = torch.max(confPred, dim=-1, keepdim=True)
            b, h, w, c = pred.size()
            oneHot = torch.zeros(b, h, w,  cfg.model.bboxPredNum).to("cuda:0").scatter_(-1, maxIndex, 1).type(torch.bool)

            coobjMask = torch.logical_and(oneHot, torch.unsqueeze(coobjMask_, -1))
            a = torch.nonzero(coobjMask , as_tuple=False)   #

            """"""
            pred = pred.to('cpu').numpy()
            a = a.to('cpu').numpy()
            dets = []
            for i in range(a.shape[0]):
                imId, hid, wid, boxid = a[i][0], a[i][1], a[i][2], a[i][3]

                fullVector = pred[imId][hid][wid]
                clsVec = fullVector[-cfg.model.clsNum:]
                boxVector = fullVector[boxid*5 : (boxid+1) * 5]
                deltax, deltay, w, h, score = boxVector[0], boxVector[1], boxVector[2], boxVector[3], boxVector[4]

                cy = (hid + deltay) * cfg.model.stride
                cx = (wid + deltax) * cfg.model.stride
                w  = w * cfg.model.netInput[1]
                h = h * cfg.model.netInput[0]

                c = np.argmax(clsVec)

                dets.append([cx, cy, w, h, score, c])

            dets = np.array(dets)
            dets[:, :2] -= dets[:, 2:4] / 2
            dets[:, 2:4] += dets[:, :2]
            dets = nms(dets, iouThresh)
            for i in range(dets.shape[0]):
                x1, y1, x2, y2, score, cls  = dets[i][0],dets[i][1], dets[i][2],dets[i][3], dets[i][4],dets[i][5],
                cv2.rectangle(image, (int(x1), int(y1)),(int(x2), int(y2)),
                              (0,0,255), 1)
                cv2.putText(image, "score: "+str(round(score,3)), (int(x1), int(y1)),1,1,(0,0,255))
                cv2.putText(image, "cls: " + str(int(cls)), (int(x1), int(y1+15)), 1, 1, (0, 0, 255))

            cv2.imshow("", image)
            cv2.waitKey()

# savePath =cfg.dir.modelSaveDir + '15.pth'
# weights = torch.load(savePath)#加载参数
# network.load_state_dict(weights)#给自己的模型加载参数
# with torch.no_grad():
#
#     for id, infos in enumerate(trainLoader):
#         """forward and pred"""
#         imgs = infos['images']
#         bboxesGt = infos['bboxesGt']
#         classesGt = infos['classes']
#         imgs = imgs.to(device).float()
#         bboxesPred, classesPred = network(imgs)
#
#         mlvl_bboxes = []
#         mlvl_scores = []
#         for level in range(headerNum):
#             feath, featw = featSizes[level]
#             clspred = classesPred[level].sigmoid()
#             boxpred = bboxesPred[level]
#             boxpred = BoxesDistribution(reg_max=bboxPredNum)(boxpred)*strides[level]
#
#             pointsx = torch.arange(0, featw, device=device).repeat(feath).reshape(feath, featw).to(device)
#             pointsy = torch.arange(0, feath, device=device).repeat(featw).reshape(featw, feath).to(device)
#             pointsy = pointsy.t()
#             points = torch.stack((pointsx, pointsy), dim=2).reshape(-1, 2).repeat(batchSize, 1, 1).reshape(-1, 2) * strides[
#                 level]
#             bboxesPredOne = distance2bbox(points, boxpred)
#
#             mlvl_bboxes.append(bboxesPredOne)
#             mlvl_scores.append(clspred.reshape(featw*feath, -1))
#         mlvl_bboxes = torch.cat(mlvl_bboxes, 0)
#         mlvl_scores = torch.cat(mlvl_scores, 0)
#
#         padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
#         mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
#         det_bboxes, det_labels = multiclass_nms(
#             mlvl_bboxes,
#             mlvl_scores,
#             score_thr=0.20,
#             nms_cfg=dict(type='nms', iou_threshold=0.4),
#             max_num=100)
#         showFlag = 1
#         if showFlag:
#             for i in range(1):
#                 image = torch.clone(imgs[i])
#                 image = image.to('cpu').numpy()#*255
#                 image = image.transpose(1, 2, 0)
#                 image = image.astype(np.uint8)
#                 image = cv2.UMat(image).get()
#                 det_bboxes = det_bboxes.to('cpu').numpy()
#                 det_labels = det_labels.to('cpu').numpy()
#
#                 for j in range(det_bboxes.shape[0]):
#                     if det_bboxes[j][-1]> 0.5:
#                         box = det_bboxes[j]
#                         cls = det_labels[j]
#                         cv2.rectangle(image, (max(0, box[0]), max(0, box[1])), (max(0, box[2]), max(0, box[3])), (0,255,0))
#                         cv2.putText(image,str(cls)+'-'+str(det_bboxes[j][-1]),
#                                     (max(0, box[0]), max(0, box[1])),1,2,(0,0,255))
#                         print(cls)
#                 #imsavePath = '/media/q/deep/me/model/pytorch_predict_' + str(id) + '.jpg'
#                 # print(imsavePath)
#                 #cv2.imwrite(imsavePath, image)
#                 cv2.imshow('img', image)
#
#                 cv2.waitKey()
#             print("show anghor box gt")

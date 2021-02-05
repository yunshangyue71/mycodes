from data.dataloader_detection_body import ListDataset
import torch
from net.net import NanoNet
from data.collate_function import collate_function
from anchor.anchorbox_generate import AnchorGenerator
from anchor.predBox_and_imageBox import distance2bbox, bbox2distance
from anchor.anchor_assigner import Assigner
from utils.bbox_distribution import BoxesDistribution
import cv2
from utils.nms import multiclass_nms
import numpy as np

batchSize = 1
headerNum = 3
featSizes = [(60,40),(30,20),(15,10)]
strides = [8,16,32]
classNum = 1
bboxPredNum = 8
netInput = (480, 320)
device = torch.device('cuda:0')

"""dataset"""
trainData = ListDataset(trainAnnoPath ='/media/q/deep/me/data/WiderPerson/Annotations/' ,# txt files root /
                        trainImgPath = '/media/q/deep/me/data/WiderPerson/Images/' , #images root /
                        netInputSizehw = netInput,
                        augFlag=0,
                        )

trainLoader = torch.utils.data.DataLoader(
    trainData,
    collate_fn=collate_function,
    batch_size=batchSize,
    shuffle=True,
    num_workers=0,
    pin_memory=True,  # 如果机器计算能力好的话，就可以设置为True，
)
network = NanoNet(classNum=classNum)
network.to(device)

savePath ='./saved_model/25.pth'
weights = torch.load(savePath)#加载参数
network.load_state_dict(weights)#给自己的模型加载参数

cap  = cv2.VideoCapture(0)
flag = 1
with torch.no_grad():
    while(flag):
        ret, img = cap.read()
        #cv2.imshow("before resize", img)
        img = cv2.resize(img, (netInput[1],netInput[0]))
        img = img.astype(np.float32) / 255
        img = img.transpose(2, 0, 1)
        imgs = img.reshape(-1, 3, netInput[0],netInput[1])

        imgs = torch.from_numpy(imgs)
        imgs = imgs.to(device).float()
        bboxesPred, classesPred = network(imgs)

        mlvl_bboxes = []
        mlvl_scores = []
        for level in range(headerNum):
            feath, featw = featSizes[level]
            clspred = classesPred[level].sigmoid()
            boxpred = bboxesPred[level]
            boxpred = BoxesDistribution(reg_max=bboxPredNum)(boxpred) * strides[level]

            pointsx = torch.arange(0, featw, device=device).repeat(feath).reshape(feath, featw).to(device)
            pointsy = torch.arange(0, feath, device=device).repeat(featw).reshape(featw, feath).to(device)
            pointsy = pointsy.t()
            points = torch.stack((pointsx, pointsy), dim=2).reshape(-1, 2).repeat(batchSize, 1, 1).reshape(-1, 2) * strides[
                level]
            bboxesPredOne = distance2bbox(points, boxpred)

            mlvl_bboxes.append(bboxesPredOne)
            mlvl_scores.append(clspred.reshape(featw * feath, -1))
        mlvl_bboxes = torch.cat(mlvl_bboxes, 0)
        mlvl_scores = torch.cat(mlvl_scores, 0)

        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            score_thr=0.10,
            nms_cfg=dict(type='nms', iou_threshold=0.4),
            max_num=100)
        showFlag = 1
        if showFlag:
            for i in range(1):
                image = torch.clone(imgs[i])
                image = image.to('cpu').numpy() * 255
                image = image.transpose(1, 2, 0)
                image = image.astype(np.uint8)
                image = cv2.UMat(image).get()
                det_bboxes = det_bboxes.to('cpu').numpy()
                det_labels = det_labels.to('cpu').numpy()

                for j in range(det_bboxes.shape[0]):
                    if det_bboxes[j][-1] > 0 or 1:
                        box = det_bboxes[j]
                        cls = det_labels[j]
                        cv2.rectangle(image, (max(0, box[0]), max(0, box[1])), (max(0, box[2]), max(0, box[3])),
                                      (0, 255, 0))
                        cv2.putText(image, str(cls) + '-' + str(det_bboxes[j][-1]),
                                    (max(0, box[0]), max(0, box[1])), 1, 2, (0, 0, 255))
                        print(cls)
                imsavePath = '/media/q/deep/me/model/pytorch_predict_' + str(id) + '.jpg'
                # print(imsavePath)
                #cv2.imwrite(imsavePath, image)
                cv2.imshow('img', image)

                cv2.waitKey(100)
            print("show anghor box gt")
        cv2.waitKey(10)


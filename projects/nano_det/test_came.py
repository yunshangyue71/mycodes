from data.dataloader_detection import ListDataset
import torch
from net.net import NanoNet
from data.collate_function import collate_function
from anchor.predBox_and_imageBox import distance2bbox
from utils.bbox_distribution import BoxesDistribution
import cv2
from utils.nms import multiclass_nms
import numpy as np
from config.config import load_config, cfg
from utils.resize_uniform import resizeUniform
from data.imgaug_wo_shape import ImgAugWithoutShape

"""config"""
load_config(cfg, "./config/config.yaml")
print(cfg)

headerNum = len(cfg.model.featSizes)
featSizes = cfg.model.featSizes
strides = cfg.model.strides
classNum = cfg.model.classNum
bboxPredNum = cfg.model.bboxPredNum
netInput = cfg.model.netInput
device = torch.device('cuda:0')
batchSize = 1

"""dataset"""
trainData = ListDataset(trainAnnoPath =cfg.dir.valAnnoDir,# txt files root /
                        trainImgPath = cfg.dir.trainImgDir , #images root /
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
network = NanoNet(classNum=cfg.model.classNum, regBoxNum=cfg.model.bboxPredNum)
network.to(device)

savePath =cfg.dir.modelSaveDir + '25.pth'
weights = torch.load(savePath)#加载参数
network.load_state_dict(weights)#给自己的模型加载参数
with torch.no_grad():
    cap = cv2.VideoCapture(0)
    while(1):
        # get a frame
        ret, frame = cap.read()
        frame, infos, bboxes = resizeUniform(frame, netInput)

        imgauger = ImgAugWithoutShape(frame)
        imgauger.normalize1(mean=np.array(cfg.data.normalize[0]), std=np.array(cfg.data.normalize[1]))
        img = imgauger.img
        img = img.transpose(2, 0, 1)  # 因为pytorch的格式是CHW
        imgs = np.array([img])
        """forward and pred"""

        imgs = torch.from_numpy(imgs).to(device).float()
        bboxesPred, classesPred = network(imgs)

        mlvl_bboxes = []
        mlvl_scores = []
        for level in range(headerNum):
            feath, featw = featSizes[level]
            clspred = classesPred[level].sigmoid()
            boxpred = bboxesPred[level]
            boxpred = BoxesDistribution(reg_max=bboxPredNum)(boxpred)*strides[level]

            pointsx = torch.arange(0, featw, device=device).repeat(feath).reshape(feath, featw).to(device)
            pointsy = torch.arange(0, feath, device=device).repeat(featw).reshape(featw, feath).to(device)
            pointsy = pointsy.t()
            points = torch.stack((pointsx, pointsy), dim=2).reshape(-1, 2).repeat(batchSize, 1, 1).reshape(-1, 2) * strides[
                level]
            bboxesPredOne = distance2bbox(points, boxpred)

            mlvl_bboxes.append(bboxesPredOne)
            mlvl_scores.append(clspred.reshape(featw*feath, -1))
        mlvl_bboxes = torch.cat(mlvl_bboxes, 0)
        mlvl_scores = torch.cat(mlvl_scores, 0)

        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            score_thr=0.40,
            nms_cfg=dict(type='nms', iou_threshold=0.2),
            max_num=100)
        showFlag = 1
        if showFlag:
            for i in range(1):
                # image = torch.clone(imgs[i])
                # image = image.to('cpu').numpy()#*255
                # image = image.transpose(1, 2, 0)
                # image = image.astype(np.uint8)
                # image = cv2.UMat(image).get()
                image = frame
                det_bboxes = det_bboxes.to('cpu').numpy()
                det_labels = det_labels.to('cpu').numpy()

                for j in range(det_bboxes.shape[0]):
                    if det_bboxes[j][-1]> 0.5:
                        box = det_bboxes[j]
                        cls = det_labels[j]
                        cv2.rectangle(image, (max(0, box[0]), max(0, box[1])), (max(0, box[2]), max(0, box[3])), (0,255,0))
                        cv2.putText(image,str(cls)+'-'+str(det_bboxes[j][-1]),
                                    (max(0, box[0]), max(0, box[1])),1,2,(0,0,255))
                        print(cls)
                #imsavePath = '/media/q/deep/me/model/pytorch_predict_' + str(id) + '.jpg'
                # print(imsavePath)
                #cv2.imwrite(imsavePath, image)
                cv2.imshow('img', image)

                cv2.waitKey(1)
            print("show anghor box gt")


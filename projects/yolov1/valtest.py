from data.dataloader_detection import ListDataset
from data.collate_function import collate_function
from config.config import load_config, cfg
from net.resnet import ResNet, ResnetBasic,ResnetBasicSlim
from utils.nms_np_simple import nms
from data.dataloader_test_detection import ListDataset as testListDataset
from data.resize_uniform import resizeUniform

from dataY.yolov1_dataY import DataY
from net.yolov1 import YOLOv1
import numpy as np
import cv2
from loss.yololoss import yoloLoss
from dataY.yolov1_dataY import DataY
import time
import math
import torch
def plotGride(img, grideHW=(7,7), stride=64):
    # plot gride
    h,w,_ = img.shape
    for li in range(grideHW[1]):
        cv2.line(img, (int(li *stride), 0),
                 (int(li * stride), int(w)), (0, 255, 0), 1)
    for li in range(grideHW[0]):
        cv2.line(img, (0, int(li * stride)),
                 (int(h), int(li * stride)), (0, 255, 0), 1)
    return img

def plotBox(img, x1, y1,x2,y2,txts):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
    for i in range(len(txts)):
        cv2.putText(img, txts[i], (int(x1), int(y1 + i*15)), 1, 1, (0, 0, 255))
    cv2.circle(img, (int((x1 +x2) / 2), int((y1 + y2) / 2)), color=(0, 0, 255), radius=2, thickness=-1)
    return img

def post(pred, dic):
    # target contain object mask,
    b, h, w, c = pred.size()

    alldets= []# [[一个图片的dets]，[],...]
    for i in range(b):
        apred = pred[i]

        """根据置信度来判断哪个网络有"""
        coobjMask1 = apred[:,:,4] > dic["scoreThresh"]
        for j in range(dic["bboxPredNum"]):
            coobjMask1 = torch.logical_or(coobjMask1, apred[:, :, 4 + i * 5] > dic["scoreThresh"])

        """每个cell 选择最大置信度的哪个"""
        confPred = apred[:, :, 4:5 * dic["bboxPredNum"]:5]
        _, maxIndex = torch.max(confPred, dim=-1, keepdim=True)
        h_, w_, c_ = apred.size()
        oneHot = torch.zeros(h_, w_, dic["bboxPredNum"]).to("cuda:0").scatter_(-1, maxIndex, 1).type(
            torch.bool)

        """最终选择"""
        coobjMask = torch.logical_and(oneHot, torch.unsqueeze(coobjMask1, -1))
        chioceIndex = torch.nonzero(coobjMask, as_tuple=False)
        chioceIndex = chioceIndex.to('cpu').numpy()

        dets = []
        for k in range(chioceIndex.shape[0]):
            hid, wid, boxid = chioceIndex[k][0], chioceIndex[k][1], chioceIndex[k][2]
            fullVector = apred[hid][wid].to("cpu").numpy()
            clsVec = fullVector[-dic["clsNum"]:]
            boxVector = fullVector[boxid * 5: (boxid + 1) * 5]
            deltax, deltay, w, h, score = boxVector[0], boxVector[1], boxVector[2], boxVector[3], boxVector[4]

            cy = (hid + deltay) * dic["stride"]
            cx = (wid + deltax) * dic["stride"]
            w = w * w * dic["netInputHw"][1]
            h = h * h * dic["netInputHw"][0]
            c = clsVec.argmax()
            dets.append([cx, cy, w, h, score, c])
        dets = np.array(dets)
        dets = dets.reshape(-1, 6)
        dets[:, :2] -= dets[:, 2:4] / 2
        dets[:, 2:4] += dets[:, :2]

        #boarder in the image
        dets[:,:2] = np.where(dets[:, :2] < 0, 0, dets[:,:2] )
        dets[:, 2] = np.where(dets[:, 2] > dic["netInputHw"][1], dic["netInputHw"][1], dets[:, 2])
        dets[:, 3] = np.where(dets[:, 3] > dic["netInputHw"][0], dic["netInputHw"][0], dets[:, 3])

        dets = nms(dets, dic["iouThresh"])
        alldets.append(dets)
    return alldets

if __name__ == '__main__':
    """config"""
    load_config(cfg, "./config/config.yaml")
    print(cfg)
    device = torch.device('cuda:0')

    batchsize = 2
    showFlag = 1
    saveFlag = 0
    saveDir = ""

    mode = 1

    # 如果是test mode 没有label 信息， 只能是预测
    # cam Mode 调用摄像头
    # val model这个有label信息，在show的时候会展示label
    modeDict = {0:"testMode", 1:"valMode", 2:"camMode"}
    postDict = {"scoreThresh": 0.3,
                "iouThresh": 0.4,
                "netInputHw":(448,448),
                "bboxPredNum": cfg.model.bboxPredNum,
                "clsNum":cfg.model.clsNum,
                "stride":cfg.model.stride,
                }

    """dataset"""
    if modeDict[mode] == "testMode":
        dataset =  testListDataset( imgPath = cfg.dir.testImgDir,  # images root /
                         netInputSizehw = cfg.model.netInput,
                         imgChannelNumber=cfg.model.imgChannelNumber,
                         clsname= cfg.clsname
                         )
    if modeDict[mode] == "valMode":
        dataset = ListDataset(trainAnnoPath=cfg.dir.valAnnoDir, trainImgPath=cfg.dir.valImgDir,
                                netInputSizehw=cfg.model.netInput, augFlag=False,
                                 clsname = cfg.clsname, imgChannelNumber=cfg.model.imgChannelNumber)
    if modeDict[mode] == "valMode" or modeDict[mode] == "testMode":
        dataLoader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_function,
            batch_size= batchsize, #cfg.train.batchSize,
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
    network = ResNet(ResnetBasicSlim,
                     # [2, 2, 2, 2],
                     [3, 4, 6, 3],
                     channel_in=cfg.data.imgChannelNumber,
                     channel_out=(cfg.model.bboxPredNum * 5 + cfg.model.clsNum))
    network = network.eval()
    # network = YOLOv1(params={"dropout": 0.5, "num_class": cfg.model.clsNum})
    network.to(device)
    weights = torch.load(cfg.dir.modelSaveDir + cfg.dir.modelName)  # 加载参数
    network.load_state_dict(weights["savedModel"])  # 给自己的模型加载参数

    with torch.no_grad():
        if modeDict[mode] == "camMode":
            cap = cv2.VideoCapture(0)

            while (1):
                assert cfg.model.imgChannelNumber == 3, "输入通道目前支支持3通道"
                ret, img = cap.read()
                frame = np.copy(img)
                if not ret:
                    print("can not cap a picture")
                    time.sleep(1)
                    continue
                img, infos = resizeUniform(img, cfg.model.netInput)

                imgs = np.array([img])
                imgs = torch.from_numpy(imgs.astype(np.float32)),
                imgs = imgs.to(device).float()
                mean = torch.tensor(cfg.data.normalize[0]).cuda().reshape(3, 1, 1)
                std = torch.tensor(cfg.data.normalize[1]).cuda().reshape(3, 1, 1)
                imgs = (imgs - mean) / std

                pred = network(imgs)
                pred = pred.permute(0, 2, 3, 1)  # rotate the BCHW to BHWC

                """post"""
                bcDets = post(pred, postDict)
                dets = bcDets[0]

                cv2.imshow("capture", img)
                """plot pred"""
                imgp = plotGride(frame, grideHW=(cfg.model.featSize), stride=cfg.model.stride)
                for i in range(dets.shape[0]):
                    x1, y1, x2, y2, score, cls = \
                        dets[i][0], dets[i][1], dets[i][2], dets[i][3], dets[i][4], dets[i][5]
                    imgp = plotBox(imgp, x1, y1, x2, y2,
                                   ["s: " + str(round(score, 3)), "c: " + cfg.clsname[cls]])
                cv2.imshow("pred", imgp)
                cv2.waitKey()
                if cv2.waitKey(1) & 0xFF == 32:
                    break


        if modeDict[mode] == "valMode" or modeDict[mode] == "testMode":
            for id, infos in enumerate(dataLoader): #每个batch
                """forward and pred"""
                imgs = infos['images']
                imgs = imgs.to(device).float()
                mean = torch.tensor(cfg.data.normalize[0]).cuda().reshape(3, 1, 1)
                std = torch.tensor(cfg.data.normalize[1]).cuda().reshape(3, 1, 1)
                imgs = (imgs - mean) / std

                pred = network(imgs)
                pred = pred.permute(0, 2, 3, 1)  # rotate the BCHW to BHWC

                """post"""
                bcDets = post(pred, postDict)

                for bcid in range(batchsize):
                    dets = bcDets[bcid]

                    if showFlag:
                        image = infos['images'][bcid]
                        image = image.to('cpu').numpy()
                        image = image.transpose(1, 2, 0).astype(np.uint8)
                        image = cv2.UMat(image).get()
                        imgt = np.copy(image)
                        imgp = np.copy(image)

                        """plot pred"""
                        imgp = plotGride(imgp, grideHW=(cfg.model.featSize), stride=cfg.model.stride)
                        for i in range(dets.shape[0]):
                            x1, y1, x2, y2, score, cls = \
                                dets[i][0], dets[i][1], dets[i][2], dets[i][3], dets[i][4], dets[i][5]
                            imgp = plotBox(imgp, x1, y1, x2, y2,
                                           [str(i)+" s: " + str(round(score, 3)), "c: " + cfg.clsname[cls]])
                            ii = math.floor(((x1+x2)/2)/cfg.model.stride)
                            jj = math.floor(((y1 + y2) / 2) / cfg.model.stride)
                            print("pred[%d x1:%.2f y1:%.2f x2:%.2f y2:%.2f w:%.2f h:%.2f score: %.2f i:%d j:%d "%(
                                i, x1,y1, x2, y2,x2-x1,y2-y1,score, ii, jj)+ cfg.clsname[cls] +"]")
                        cv2.imshow("pred", imgp)
                        cv2.waitKey()

                    if showFlag and not modeDict[mode] == "testModeFlag":# test mode 没有target
                        """read target"""
                        bboxesGt = infos['bboxesGt'][bcid]
                        classesGt = infos['classes'][bcid]
                        annoName = infos["annoName"][bcid]

                        """plot target"""
                        imgt = plotGride(imgt, grideHW=(cfg.model.featSize), stride=cfg.model.stride)
                        for i in range(bboxesGt.shape[0]):
                            x1, y1, w, h =  bboxesGt[i]
                            cls = classesGt[i]
                            plotBox(imgt, x1,y1,x1+w,y1+h,[cfg.clsname[cls]])
                            ii = math.floor((x1 + w/ 2) / cfg.model.stride)
                            jj = math.floor((y1 + h / 2) / cfg.model.stride)
                            print("target[x1:%.2f y1:%.2f x2:%.2f y2:%.2f w:%.2f h:%.2f i:%d j:%d " % (
                                x1, y1,x1+w+1,y1+h+1, w, h, ii, jj)+ cfg.clsname[cls]+"]")
                        print(annoName)
                        cv2.imshow("target", imgt)
                        cv2.waitKey()
                        print("-"*50)


                    if saveFlag and  modeDict[mode] == "valModeFlag":
                        np.savetxt(saveDir + infos["annoName"][bcid],dets)
                    if saveFlag and modeDict[mode] == "testModeFlag":
                        np.savetxt(saveDir + infos["imgName"][bcid].split["."][0]+".txt",dets)
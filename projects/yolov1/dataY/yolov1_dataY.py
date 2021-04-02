"""
处理dataY， 使之能够用于损失计算
处理一张图片
方式可以改写为TF、torch等框架加速计算，目前是纯cpu的
"""
import numpy as np
import torch

class DataY(object):
    def __init__(self,
                 inputHW=(448, 448),  # 指定了inputsize 这是因为输入的是经过resize后的图片
                 gride = (7,7), # 将网络输入成了多少个网格
                 stride = 64,
                 boxNum=2,
                 clsNum = 10,
                 device = "cuda:0"):
        self.boxNum = boxNum
        self.clsNum = clsNum

        self.gride = gride
        self.inputHW = torch.from_numpy(np.array(inputHW)).to(device)
        self.stride = stride
        assert inputHW[0] / gride[0] == stride and inputHW[1] / gride[1] == stride, "please check the shape"

        self.device = device
    #box:B, (x1, y1, w, h)
    #cls:(B, N)
    def do(self, boxes, clses):
        b = len(boxes)
        target = []
        for k in range(b): # 遍历batch
            box = torch.from_numpy(boxes[k]).to(self.device)
            cls = torch.from_numpy(clses[k]).to(self.device)

            atarget = torch.zeros(self.gride[0], self.gride[1], 5 * self.boxNum + self.clsNum)
            cxcy = box[:, :2] + box[:, 2:] / 2
            wh = box[:, 2:]
            for i in range(cxcy.size()[0]):# 遍历每个图片的box
                cxcyi = cxcy[i]
                whi = wh[i]
                ij = (cxcyi / self.stride).ceil() - 1   # 判断在哪一个gride 里面  # 必须使得stride 是wh， 否则输入wh不相等会有问题
                xy = cxcyi / self.stride - ij           # 距离gride 左上角的距离， gride 表示单位长度
                whi = whi / self.inputHW                # 物体宽度是图片的比例
                clsi = cls[i]


                for j in range(self.boxNum):
                    atarget[int(ij[1]), int(ij[0]), (0 + 5 * j) : (1 + 5 * j + 1)] = xy
                    atarget[int(ij[1]), int(ij[0]), (2 + 5 * j) : (3 + 5 * j + 1)] = whi
                    atarget[int(ij[1]), int(ij[0]), 4 + 5 * j] = 1

                atarget[int(ij[1]), int(ij[0]), int(clsi + 5 * self.boxNum)] = torch.ones(1)
            target.append(atarget)
        target = torch.stack(target, dim=0)

        return target.to(self.device)

    def do2(self, boxes, clses, preds):
        preds = preds.permute(0, 2, 3, 1)
        b = len(boxes)
        target = []
        for k in range(b):  # 遍历batch
            box = torch.from_numpy(boxes[k]).to(self.device)
            cls = torch.from_numpy(clses[k]).to(self.device)
            pred = preds[k]

            atarget = torch.zeros(self.gride[0], self.gride[1], 5 * self.boxNum + self.clsNum)
            cxcy = box[:, :2] + box[:, 2:] / 2
            wh = box[:, 2:]
            for i in range(cxcy.size()[0]):  # 遍历每个图片的box
                cxcyi = cxcy[i]
                whi = wh[i]
                ij = (cxcyi / self.stride).ceil() - 1  # 判断在哪一个gride 里面  # 必须使得stride 是wh， 否则输入wh不相等会有问题
                xy = cxcyi / self.stride - ij  # 距离gride 左上角的距离， gride 表示单位长度
                whi = whi / self.inputHW  # 物体宽度是图片的比例
                clsi = cls[i]

                # 计算pred 和这个 box的 iou， 看看和那个pred box iou最大
                x1t = box[i][0]
                y1t = box[i][1]
                x2t = box[i][0] + box[i][2]
                y2t = box[i][1] + box[i][3]


                choiceBoxId = 0 #每个box 在pred多个box里面只有iou最大的那个才是针织
                maxIou = 0
                for j in range(self.boxNum):
                    try:
                        xywhc  = pred[int(ij[1]),int(ij[0]),j*5:j*5+5]
                    except:
                        print("")
                    cxp = (int(ij[0]) + xywhc[0])*self.stride
                    cyp = (int(ij[1] )+ xywhc[1])*self.stride
                    wp = xywhc[2] * self.inputHW[1]
                    hp = xywhc[3] * self.inputHW[0]
                    x1p = cxp - wp/2
                    y1p = cyp - hp/2
                    x2p = cxp + wp/2
                    y2p = cyp + hp/2

                    union = max((x2t-x1t),0)*max((y2t-y1t),0) + max((y2p-y1p), 0) * max((x2p-x1p), 0)
                    #max(max(x2p, x2t) - min(x1p, x1t), 0)* max(max(y2p, y2t) - min(y1p, y1t), 0)

                    inter = max(min(x2p, x2t) - max(x1p, x1t), 0)* max(min(y2p, y2t) - max(y1p, y1t),0)
                    iou = inter/(union-inter)
                    if iou > maxIou:
                        maxIou = iou
                        choiceBoxId = j

                #看看对应cell最大的置信度是多少， 如果超过他那么就替换他， 如果没有超过就舍弃这个box
                maxIoubefore = 0
                for j in range(self.boxNum):
                    if atarget[int(ij[1]), int(ij[0]), 4 + 5 * j] > maxIoubefore:
                        maxIoubefore = atarget[int(ij[1]), int(ij[0]), 4 + 5 * j]
                if maxIou > maxIoubefore:
                    atarget[int(ij[1]), int(ij[0]), (0 + 5 * choiceBoxId): (1 + 5 * choiceBoxId + 1)] = xy
                    atarget[int(ij[1]), int(ij[0]), (2 + 5 * choiceBoxId): (3 + 5 * choiceBoxId + 1)] = whi
                    atarget[int(ij[1]), int(ij[0]), 4 + 5 * choiceBoxId] = maxIou

                    atarget[int(ij[1]), int(ij[0]), int(clsi + 5 * self.boxNum)] = maxIou
            target.append(atarget)
        target = torch.stack(target, dim=0)

        return target.to(self.device)
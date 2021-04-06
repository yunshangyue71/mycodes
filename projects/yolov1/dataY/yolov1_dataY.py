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
        self.inputHW = np.array(inputHW)
        self.stride = stride
        assert inputHW[0] / gride[0] == stride and inputHW[1] / gride[1] == stride, "please check the shape"
        self.device = device

    #box:B, (x1, y1, w, h)
    #cls:(B, N)
    def do2(self, boxes, clses, preds):
        preds = preds.permute(0, 2, 3, 1)
        target = []
        for k in range(len(boxes)):  # 遍历batch 的每个图片
            atarget = torch.zeros(self.gride[0], self.gride[1], 5 * self.boxNum + self.clsNum)
            if (boxes[k]==np.array([[-1,-1,-1,-1]])).all():
                target.append(atarget)
                continue
            # boxt = torch.from_numpy(boxes[k]).to(self.device)
            # clst = torch.from_numpy(clses[k]).to(self.device)
            # cxcyt = boxt[:, :2] + boxt[:, 2:] / 2
            # wht = boxt[:, 2:]
            # boxxyxyt = boxt[:, :2] + boxt[:, 2:]
            # pred = preds[k]

            for i in range(boxes[k].shape[0]):  # 遍历图片的每个box, 看这个box 落在哪里
                boxt = boxes[k][i]
                boxxyxyt = np.copy(boxt)
                boxxyxyt[2:] += boxxyxyt[:2]
                cxcyt = boxt[:2] + boxt[2:]/2
                wht = boxt[2:]

                #用于寻找 loacte 以及制作vector
                ij = np.ceil(cxcyt / self.stride) - 1  # 判断在哪一个gride 里面  # 必须使得stride 是wh， 否则输入wh不相等会有问题
                xy = cxcyt / self.stride - ij  # 距离gride 左上角的距离， gride 表示单位长度
                wht = wht / self.inputHW[::-1]  # 物体宽度是图片的比例
                clst = clses[k][i]

                # 计算pred 和这个 box的 iou， 看看和那个pred box iou最大
                x1t = boxxyxyt[0]
                y1t = boxxyxyt[1]
                x2t = boxxyxyt[2]
                y2t = boxxyxyt[3]

                """判断这个box 可以locate在哪里，
                判断逻辑是：
                这个box 和pred 多个box 里面哪个box的iou最大， 选中他
                """
                choiceBoxId = 0 #每个box 在pred多个box里面只有iou最大的那个才是针织
                maxIou = 0
                for j in range(self.boxNum):
                    xywhc  = preds[k][int(ij[1]),int(ij[0]),j*5:j*5+5]
                    cxp = (int(ij[0]) + xywhc[0])*self.stride
                    cyp = (int(ij[1]) + xywhc[1])*self.stride
                    wp = xywhc[2] * self.inputHW[1]
                    hp = xywhc[3] * self.inputHW[0]
                    x1p = cxp - wp/2
                    y1p = cyp - hp/2
                    x2p = cxp + wp/2
                    y2p = cyp + hp/2

                    union = max((x2t-x1t),0)*max((y2t-y1t),0) + max((y2p-y1p), 0) * max((x2p-x1p), 0)

                    inter = max(min(x2p, x2t) - max(x1p, x1t), 0)* max(min(y2p, y2t) - max(y1p, y1t),0)
                    iou = inter / (union-inter)
                    if iou > maxIou:
                        maxIou = iou
                        choiceBoxId = j

                """
                图片的这个box 和 第choiceBoxId个pred box的 iou 是否比之前的大， 这样依然可能是两个
                """
                maxIoubefore = 0
                for j in range(self.boxNum):
                    if atarget[int(ij[1]), int(ij[0]), 4 + 5 * j] > maxIoubefore:
                        maxIoubefore = atarget[int(ij[1]), int(ij[0]), 4 + 5 * j]
                if maxIou > maxIoubefore:
                    atarget[int(ij[1]), int(ij[0]), (0 + 5 * choiceBoxId): (1 + 5 * choiceBoxId + 1)] = torch.from_numpy(xy).to(self.device)
                    atarget[int(ij[1]), int(ij[0]), (2 + 5 * choiceBoxId): (3 + 5 * choiceBoxId + 1)] = torch.from_numpy(wht).to(self.device)

                    atarget[int(ij[1]), int(ij[0]), 4 + 5 * choiceBoxId] = maxIou

                    atarget[int(ij[1]), int(ij[0]), int(clst + 5 * self.boxNum)] = maxIou
            target.append(atarget)
        target = torch.stack(target, dim=0)

        return target.to(self.device)
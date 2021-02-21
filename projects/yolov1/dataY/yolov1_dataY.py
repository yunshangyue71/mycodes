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
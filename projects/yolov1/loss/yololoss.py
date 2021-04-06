#yololoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class yoloLoss(object):
    def __init__(self,
                 boxNum=2,
                 clsNum=10,
                 lsNoObj = 0.05,
                 lsObj=1,
                 lsConf = 1,
                 lsCls = 1,
                 lsBox = 1):
        self.boxNum = boxNum
        self.clsNum = clsNum

        self.lsNoObj = lsNoObj
        self.lsObj = lsObj

        self.lsConf = lsConf
        self.lsBox = lsBox
        self.lsCls =  lsCls

    def do(self, pred, target):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        '''
        pred = pred.permute(0, 2, 3, 1)       # rotate the BCHW to BHWC
        coobjMask_ = target[:, :, :, 4] > 0    # target contain object mask

        #选择每个anchor 点 预测置信度最大的, bbox num = 1 代码也可以
        confPred = pred[:, :, :, 4:5 * self.boxNum:5]
        maxNum, maxIndex = torch.max(confPred, dim=-1, keepdim=True)
        b, h, w, c = pred.size()
        oneHot = torch.zeros(b, h, w, self.boxNum).to("cuda:0").scatter_(-1, maxIndex, 1).type(torch.bool)

        coobjMask = torch.logical_and(oneHot, torch.unsqueeze(coobjMask_, -1))
        noobjMask = torch.logical_not(coobjMask)   # no object mask

        """confidence loss"""
        confPred = pred[:,:,:,4:5 * self.boxNum :5]
        confTarget = target[:, :, :, 4:5 * self.boxNum:5]
        ls = torch.pow((confPred - confTarget), 2)
        lsConf =  ls * coobjMask * self.lsObj  +ls * noobjMask * self.lsNoObj  # 假设全部都是noobj

        """cls loss"""
        clsPred = pred[:,:,:,-self.clsNum:]
        clsTarget = target[:,:,:, -self.clsNum:]
        ls = torch.pow((clsPred - clsTarget), 2)
        lsCls = ls * coobjMask_.unsqueeze(-1) * self.lsCls

        """bbox loss"""
        xPred =     pred[:, :, :, 0:(0 + 5 * self.boxNum):5]
        xTarget = target[:, :, :, 0:(0 + 5 * self.boxNum):5]
        yPred =     pred[:, :, :, 1:(1 + 5 * self.boxNum):5]
        yTarget = target[:, :, :, 1:(1 + 5 * self.boxNum):5]
        wPred =     pred[:, :, :, 2:(2 + 5 * self.boxNum):5]+10**-8
        wTarget = target[:, :, :, 2:(2 + 5 * self.boxNum):5]+10**-8
        hPred =     pred[:, :, :, 3:(3 + 5 * self.boxNum):5]+10**-8
        x = float(hPred[0][0][0][0])
        if math.isnan(x):
            print("nn--------------",hPred)
            raise
            #print("nn--------------", wPred)
        # else:
            # print("nn",hPred)
            # print("nn", wPred)
        hTarget = target[:, :, :, 3:(3 + 5 * self.boxNum):5]+10**-8
        lsX = (xPred - xTarget).pow(2)
        lsY = (yPred - yTarget).pow(2)
        lsW = (wPred - wTarget.sqrt()).pow(2)
        lsH = (hPred - hTarget.sqrt()).pow(2)
        ls =  lsX + lsY + lsW + lsH
        # lsBox = ls * noobjMask * self.lsNoObj + ls * coobjMask * self.lsObj
        lsBox = ls * coobjMask * self.lsObj
        # #a =  lsBox.sum()
        # #loss = lsConf.sum() * self.lsConf+ lsCls.sum()*self.lsCls + lsBox.sum()*self.lsBox
        # loss = lsConf.sum() * self.lsConf +  lsBox.sum() * self.lsBox

        info = {"conf": lsConf.sum(),
                "cls": lsCls.sum(),
                "box": lsBox.sum()}
        return info
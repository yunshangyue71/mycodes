import torch.nn as nn
import torch.nn.functional as F
import torch
from backbone_block import basic_cnn
from utils_frequent.init_net import xavier_init
from easydict import EasyDict as Edict

"""
inputs:
    输入的一个特征图 
可以修改的地方、可以加上选项 
bottomUp feature map 到 topDown feature map 的卷积形式
topDown 下采样的方式，目前是插值

"""

cfg = Edict()
cfg.featureMapNum = 3   #总共多少特征图输入， 经过几层
cfg.bottomUpChannelsNum = [58, 116,232,464]#bootomUp每层的通道数， 第0个是输入特征图的featuremap， 后面几层是金字塔
cfg.topDownChannelsNum = 96
cfg.outChannelsNum = [96,96,96]      #FPN每层输出的channel 数目

class FPN(nn.Module):
    def __init__(self, cfg):
        super(FPN, self).__init__()
        self.featureMapNum = cfg.featureMapNum
        self.bottomUpChannelsNum= cfg.bottomUpChannelsNum
        self.outChannelsNum = cfg.outChannelsNum
        self.topDownChannelsNum = cfg.topDownChannelsNum

        assert self.featureMapNum == len(self.bottomUpChannelsNum)-1  == len(self.outChannelsNum)
        assert isinstance(self.topDownChannelsNum, int)

        self.makeLayers()
        self.init_weight()

    def makeLayers(self):
        #bottom up
        self.moduleListBottomUp = nn.ModuleList()
        for i in range(self.featureMapNum):
            conv = basic_cnn.ConvBnReluPool(self.bottomUpChannelsNum[i], self.bottomUpChannelsNum[i + 1], kernelSize=3, stride=1,
                                            bias = True, bn=True, relu=True, maxp2=True)
            self.moduleListBottomUp.append(conv)

       #bottomup - topdown
        self.moduleListBottomUp2TopDown = nn.ModuleList()
        for i in range(self.featureMapNum):
            conv = basic_cnn.ConvBnReluPool(self.bottomUpChannelsNum[i + 1], self.topDownChannelsNum, kernelSize=3, stride=1,
                                            bias=True, bn=True, relu=True, maxp2=False)
            self.moduleListBottomUp2TopDown.append(conv)

        #topdown - outputs
        self.moduleListOut = nn.ModuleList()
        for i in range(self.featureMapNum):
            conv = basic_cnn.ConvBnReluPool(self.topDownChannelsNum, self.outChannelsNum[i], kernelSize=3, stride=1,
                                            bias=True, bn=True, relu=True, maxp2=False)
            self.moduleListOut.append(conv)

    def init_weight(self):
        for conv in self.modules():
            if isinstance(conv, nn.Conv2d):
                xavier_init(conv, distribution='uniform')

    def forward(self, inputs):
        x = inputs

        #bottom up
        bottomUpFeat = []
        for i in range(self.featureMapNum):
            x = self.moduleListBottomUp[i](x)
            bottomUpFeat.append(x)

        #bottomup - topdown
        bottomUp2TopDownFeat = []
        for i in range(self.featureMapNum):
            feat = self.moduleListBottomUp2TopDown[i](bottomUpFeat[i])
            bottomUp2TopDownFeat.append(feat)

        #topdown
        topDownFeat = [bottomUp2TopDownFeat[-1]]
        for i in range(1, self.featureMapNum):
            feat1 = bottomUp2TopDownFeat[-(i + 1)]

            feat0 = topDownFeat[i - 1]
            downsampleSize = feat1.shape[2:]
            feat0 = F.interpolate(feat0, size=downsampleSize, mode='bilinear')

            feat = feat0+feat1
            topDownFeat.append(feat)

        #outputs
        outputs = []
        for i in range(self.featureMapNum):
            feat = self.moduleListOut[i](topDownFeat[i])
            outputs.append(feat)
        outputs.reverse()
        return tuple(outputs)
        # return outputs[0]
# 输入是bottom up 输出也是bottom up的格式
class nanodet_PAN(nn.Module):
    def __init__(self,
                 inputChanNum = [116, 232, 464],
                 chanNum = 96 # 讲输入统一到 多少个 channel
                 ):
        super(nanodet_PAN, self).__init__()
        self.chanNum = chanNum
        self.inputChanNum = inputChanNum
        self.__convBlock()
        # FPN 这里没有用卷积， 所以这里不用家
        self.init_weight()

    def __convBlock(self):
        self.conModelList = nn.ModuleList()
        for i in range(len(self.inputChanNum)):
            c = self.inputChanNum[i]
            ml = nn.Sequential(
                nn.Conv2d(c, out_channels=self.chanNum, kernel_size=1, stride=1, padding=0,
                                  groups=1, bias=False),
                nn.ReLU()
            )
            self.conModelList.append(ml)

    def makeLayers(self, x): # x:(0 ——>-1); 这个不是topdown和bottomup的意思， 是从0开始处理的意思
        le = len(x)
        y = []
        y.append(x[0])
        for i in range(1, le):
            if x[0].size()[-1] < x[-1].size()[-1]:  #top down
                a = F.interpolate(y[i - 1], size=list(x[i].size()[-2:]), mode='bilinear') # 改这里就可以改成不同的FPN
            else:
                a = F.interpolate(y[i - 1], size=list(x[i].size()[-2:]), mode='bilinear')
            b = x[i]
            y.append(a + b)
        return y

    def init_weight(self):
        for conv in self.modules():
            if isinstance(conv, nn.Conv2d):
                xavier_init(conv, distribution='uniform')

    def forward(self, x):# x(bottom up)
        y = []
        for i in range(len(x)):
            xi = x[i]
            xi = self.conModelList[i](xi)
            y.append(xi)

        y.reverse()                 # top down
        y = self.makeLayers(y)
        y.reverse()                 # bottom up
        y = self.makeLayers(y)
        return tuple(y)

if __name__ == '__main__':
    net = FPN(cfg)
    x = torch.rand(2,58,320,320)
    net(x)
    onnx_path = '/media/q/deep/me/model/pytorch_script_use/FPN.onnx'
    import netron
    torch.onnx.export(net, x, onnx_path)
    netron.start(onnx_path)

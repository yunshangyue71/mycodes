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
cfg.bottomUp2ChannelsNum = 96
cfg.outChannelsNum = [96,96,96]      #FPN每层输出的channel 数目

class PAN(nn.Module):
    def __init__(self, cfg):
        super(PAN, self).__init__()
        self.featureMapNum = cfg.featureMapNum

        self.bottomUpChannelsNum= cfg.bottomUpChannelsNum
        self.topDownChannelsNum = cfg.topDownChannelsNum
        self.bottomUp2ChannelsNum = cfg.bottomUp2ChannelsNum
        self.outChannelsNum = cfg.outChannelsNum

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

        #topdown - bottomup
        self.moduleListtopDown2BottomUp = nn.ModuleList()
        for i in range(self.featureMapNum):
            conv = basic_cnn.ConvBnReluPool(self.topDownChannelsNum, self.bottomUp2ChannelsNum, kernelSize=3, stride=1,
                                            bias=True, bn=True, relu=True, maxp2=False)
            self.moduleListtopDown2BottomUp.append(conv)

        #outputs
        self.moduleListOut = nn.ModuleList()
        for i in range(self.featureMapNum):
            conv = basic_cnn.ConvBnReluPool(self.bottomUp2ChannelsNum, self.outChannelsNum[i], kernelSize=3, stride=1,
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

            feat0 = topDownFeat[i-1]
            downsampleSize = feat1.shape[2:]
            feat0 = F.interpolate(feat0, size=downsampleSize, mode='bilinear')

            feat = feat0+feat1
            topDownFeat.append(feat)

        #topdown - bottomup2
        topDown2BottomUp2Feat = []
        for i in range(self.featureMapNum):
            feat = self.moduleListtopDown2BottomUp[i](topDownFeat[i])
            topDown2BottomUp2Feat.append(feat)

        #bottomup2
        bottomUp2Feat = [topDown2BottomUp2Feat[-1]]
        for i in range(1, self.featureMapNum):
            feat1 = topDown2BottomUp2Feat[-(i + 1)]

            feat0 = bottomUp2Feat[i - 1]
            downsampleSize = feat1.shape[2:]
            feat0 = F.interpolate(feat0, size=downsampleSize, mode='bilinear')

            feat = feat0 + feat1
            bottomUp2Feat.append(feat)

        #outputs
        outputs = []
        for i in range(self.featureMapNum):
            feat = self.moduleListOut[i](bottomUp2Feat[i])
            outputs.append(feat)
        return tuple(outputs)


if __name__ == '__main__':
    net = PAN(cfg)

    x = torch.rand(2,58,320,320)
    net(x)
    onnx_path = '/media/q/deep/me/model/pytorch_script_use/PAN.onnx'
    import netron
    torch.onnx.export(net, x, onnx_path)
    netron.start(onnx_path)

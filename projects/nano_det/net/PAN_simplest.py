import torch.nn as nn
import torch.nn.functional as F
from net import basic_cnn
from net.init_net import xavier_init

"""
inputs:[[bottom][up]]
    输入的一个多层的特征图 
可以修改的地方、可以加上选项 
bottomUp feature map 到 topDown feature map 的卷积形式
topDown 下采样的方式，目前是插值
"""

class nanodet_PAN(nn.Module):
    def __init__(self,
                 featureMapNum=3,                           #总共多少特征图输入， 经过几层
                 bottomUpChannelsNum =[116,232,464],        #foward inputs bootomUp每层的通道数，
                 topDownChannelsNum = 96,                   #因为通道的合并拼接，所以这里channel数目一般都不变
                 bottomUp2ChannelsNum = 96,
                 outChannelsNum = 96                         #FPN每层输出的channel 数目
                 ):
        super(nanodet_PAN, self).__init__()
        self.featureMapNum = featureMapNum

        self.bottomUpChannelsNum= bottomUpChannelsNum
        self.topDownChannelsNum = topDownChannelsNum
        self.bottomUp2ChannelsNum = bottomUp2ChannelsNum
        self.outChannelsNum = outChannelsNum

        assert self.featureMapNum == len(self.bottomUpChannelsNum)
        assert isinstance(self.topDownChannelsNum, int)

        self.makeLayers()
        self.init_weight()

    def makeLayers(self):
        #bottomup - topdown
        self.moduleListBottomUp2TopDown = nn.ModuleList()
        for i in range(self.featureMapNum):
            conv = basic_cnn.ConvBnReluPool(self.bottomUpChannelsNum[i], self.topDownChannelsNum, kernelSize=1, stride=1,
                                            bias=True, bn=True, relu=True, maxp2=False)
            self.moduleListBottomUp2TopDown.append(conv)

    def init_weight(self):
        for conv in self.modules():
            if isinstance(conv, nn.Conv2d):
                xavier_init(conv, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.featureMapNum

        bottomUpFeat = inputs

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
        topDown2BottomUp2Feat = topDownFeat

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
        outputs = bottomUp2Feat

        return tuple(outputs)


if __name__ == '__main__':
    pass
    # from torchsummary import summary
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = nanodet_PAN(cfg).to(device)
    # summary(net, (58, 320, 320))


    # net = nanodet_PAN(cfg)
    # import netron
    # import os
    #
    # x = torch.rand(2,58,320,320)
    # net(x)
    # name = os.path.basename(__file__)
    # name = name.split('.')[0]
    # onnx_path = '/media/q/deep/me/model/pytorch_script_use/'+name+'.onnx'
    # torch.onnx.export(net, x, onnx_path)
    # netron.start(onnx_path)

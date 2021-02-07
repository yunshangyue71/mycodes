from net.shuffleNetV2 import ShuffleNetV2
from net.PAN_simplest import nanodet_PAN
from net.header import Head
from torch import nn

from easydict import EasyDict as Edict
cfgMe = Edict()
cfgMe.modelSize = '1.0x'
cfgMe.activation = 'LeakReLu'

class NanoNet(nn.Module):
    def __init__(self, classNum):
        super(NanoNet, self).__init__()
        self.backbone = ShuffleNetV2(model_size='1.0x',activation='LeakyReLU')
        self.neck = nanodet_PAN(featureMapNum=3,
                                bottomUpChannelsNum =[116,232,464],
                                topDownChannelsNum = 96,
                                bottomUp2ChannelsNum = 96,
                                outChannelsNum = 96
                                )
        self.clsNum = classNum
        self.head = nn.ModuleList()
        for i in range(3):
            h = Head(reg_max = 8,  # defalut =8个bbox,用于分布, general focal loss format
                    inChannels = 96, #
                    clsOutChannels = self.clsNum)

            self.head.append(h)

    def forward(self, x):
        xs = self.backbone(x)
        xs = self.neck(xs)
        outs = []
        for i in range(3):
            outs.append(self.head[i](xs[i]))

        #网络输出就要是自己想要的格式， 不然torch的permute和numpy以及自己部署中的可能不一致，引发误差
        classes = []
        bboxes = []
        for level in range(3):
            pred = outs[level]
            classes.append(pred[:, :self.clsNum, :, :].permute(0,2,3,1))
            bboxes.append(pred[:, self.clsNum:, :, :].permute(0,2,3,1))
        return tuple(bboxes), tuple(classes)
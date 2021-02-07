import torch
from torch import nn
from net.init_net import xavier_init
from net.basic_cnn import DWConvBnReluPool
"""
DW-DW-PW
"""

class Head(nn.Module):
    def __init__(self,reg_max = 8, #defalut =8个bbox,用于分布, general focal loss format
                 inChannels = 96, #
                 clsOutChannels = 7):
        super(Head, self).__init__()
        self.reg_max = reg_max
        self.inChannels = inChannels
        self.clsOutChannels = clsOutChannels
        self.makeLayers()

    def makeLayers(self):
        self.head= nn.ModuleList()
        for i in range(2):
            conv = DWConvBnReluPool(self.inChannels,self.inChannels, kernelSize = 3, stride = 1,
                                    bias = True, bn = True, relu = True, maxp2 = False)
            self.head.append(conv)

        conv = nn.Conv2d(self.inChannels,
                         self.clsOutChannels + 4 * (self.reg_max),
                        1)
        self.head.append(conv)
    def init_weight(self):
        for conv in self.modules():
            if isinstance(conv, nn.Conv2d):
                xavier_init(conv, distribution='uniform')
    def forward(self, x):
        for conv in self.head:
            x = conv(x)
        return x

if __name__ == '__main__':

    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Head().to(device)
    summary(net, (96, 320, 320))


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

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnReluPool(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize=3, stride=1, bias = True, bn=False, relu=True, maxp2=False):
        super(ConvBnReluPool, self).__init__()
        self.inChannels = inChannels
        self.conv = nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding=(kernelSize-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        self.maxp2 = maxp2
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        #print(x.size())
        assert x.size()[1] == self.inChannels, "{} {}".format(x.size()[1], self.inChannels)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.maxp2:
            x = F.max_pool2d(x, kernel_size = 2, stride = 2)
        return x

class DWConvBnReluPool(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize=3, stride=1, bias=True, bn=False, relu=True,
                 maxp2=False):
        super(DWConvBnReluPool, self).__init__()
        assert inChannels==outChannels
        self.inChannels = inChannels
        self.conv = nn.Conv2d(inChannels,inChannels, kernelSize, stride,  padding=(kernelSize - 1) // 2,
                  groups=inChannels, bias=bias)
        #self.conv = nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding=(kernelSize - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        self.maxp2 = maxp2
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        #print(x.size())
        assert x.size()[1] == self.inChannels, "{} {}".format(x.size()[1], self.inChannels)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.maxp2:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x

if __name__ == '__main__':
    net = ConvBnReluPool(inChannels=10, outChannels=10, kernelSize=3, stride=1, bias = True, bn=False, relu=True, maxp2=False).cuda()

    torch.save(net.state_dict(), 'FPN.pt')
    #torchsummary.summary(net, (58, 320, 320))
    #print(net)
    import netron

    modelpath = 'FPNs.pt'
    netron.start(modelpath)
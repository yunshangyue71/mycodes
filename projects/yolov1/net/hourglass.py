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

class Number(nn.Module):
    def __init__(self, inc, outc):
        super(Number, self).__init__()
        self.conv0 = ConvBnReluPool(inChannels=inc, outChannels=16, kernelSize=7, stride=1, bias=True, bn=True,
                                    relu=True, maxp2=True)
        self.conv1 = ConvBnReluPool(inChannels=16, outChannels=32, kernelSize=3, stride=1, bias = True, bn=True, relu=True, maxp2=True)
        self.conv21 = ConvBnReluPool(inChannels=32, outChannels=64, kernelSize=3, stride=1, bias=True, bn=True,
                                    relu=True, maxp2=True)
        self.conv22_1 = ConvBnReluPool(inChannels=32, outChannels=64, kernelSize=3, stride=1, bias=True, bn=True,
                                    relu=True, maxp2=True)

        self.conv31 = ConvBnReluPool(inChannels=64, outChannels=128, kernelSize=3, stride=1, bias=True, bn=True,
                                    relu=True, maxp2=True)
        self.conv32_1 = ConvBnReluPool(inChannels=64, outChannels=128, kernelSize=3, stride=1, bias=True, bn=True,
                                    relu=True, maxp2=True)
        self.conv32_2 = ConvBnReluPool(inChannels=128, outChannels=128, kernelSize=3, stride=1, bias=True, bn=True,
                                     relu=True, maxp2=False)
        self.conv4 = ConvBnReluPool(inChannels=128, outChannels=outc, kernelSize=3, stride=1, bias=True, bn=False,
                                    relu=True, maxp2=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        y0 = self.conv0(x)
        y1 = self.conv1(y0)

        y21 = self.conv21(y1)
        y22 = self.conv22_1(y1)

        y2 = y21+ y22

        y31 = self.conv31(y2)
        y32 = self.conv32_1(y2)

        y = y31+y32
        y = self.conv4(y)
        return torch.sigmoid(y)

if __name__ == '__main__':
    net = ConvBnReluPool(inChannels=10, outChannels=10, kernelSize=3, stride=1, bias = True, bn=False, relu=True, maxp2=False).cuda()

    torch.save(net.state_dict(), 'FPN.pt')
    #torchsummary.summary(net, (58, 320, 320))
    #print(net)
    import netron

    modelpath = 'FPNs.pt'
    netron.start(modelpath)
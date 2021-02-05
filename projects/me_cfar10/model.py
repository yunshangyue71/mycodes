import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnReluPool(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize=3, stride=1, bias = True, bn=False, relu=True, maxp2=False):
        super(ConvBnReluPool, self).__init__()
        self.inChannels = inChannels
        self.conv = nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding=(kernelSize - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        self.maxp = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(outChannels)
        if maxp2 == True:
            self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        assert x.size()[1] == self.inChannels, "{} {}".format(x.size()[1], self.inChannels)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.maxp is not None:
            x = self.maxp(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ConvBnReluPool(inChannels=3,outChannels=64,kernelSize=3, stride = 1,
                                    bias = False, bn = True, relu=True, maxp2=False)
        self.conv2 = ConvBnReluPool(inChannels=64, outChannels=64, kernelSize=3, stride=1,
                                    bias = False, bn=True, relu=True, maxp2=True)
        self.conv3 = ConvBnReluPool(inChannels=64, outChannels=128, kernelSize=3, stride=1,
                                    bias = False, bn=True, relu=True, maxp2=False)
        self.conv4 = ConvBnReluPool(inChannels=128, outChannels=128, kernelSize=3, stride=1,
                                    bias = False, bn=True, relu=True, maxp2=True)
        self.conv5 = ConvBnReluPool(inChannels=128, outChannels=256, kernelSize=3, stride=1,
                                    bias = False, bn=True, relu=True, maxp2=False)
        self.conv6 = ConvBnReluPool(inChannels=256, outChannels=256, kernelSize=3, stride=1,
                                    bias = False, bn=True, relu=True, maxp2=True)


        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))  # 自适应池化，指定池化输出尺寸为 1 * 1
        self.fc1 = ConvBnReluPool(inChannels=256, outChannels=512, kernelSize=1, stride=1,
                                    bias = True, bn=False, relu=True, maxp2=False)
        self.fc2 = ConvBnReluPool(inChannels=512, outChannels=10, kernelSize=1, stride=1,
                                  bias=True, bn=False, relu=False, maxp2=False)

        self.mlist = nn.ModuleList()
        self.mlist.append(self.conv2)
        self.mlist.append(self.conv3)
        self.mlist.append(self.conv4)
        self.mlist.append(self.conv5)
    def forward(self, x):
        out = self.conv1(x)
        for conv in self.mlist:
            out = conv(out)

        # out = self.conv2(out)
        # out = self.conv3(out)
        # out = self.conv4(out)
        # out = self.conv5(out)
        out = self.conv6(out)
        out = self.global_avg_pool(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np

RELU = nn.ReLU(inplace=True)

def bnrelu(outnum):
    return nn.Sequential(nn.BatchNorm2d(outnum),
                         nn.ReLU(inplace=False))

def gnrelu(outnum):
    return nn.Sequential(nn.GroupNorm(16, outnum),
                         nn.ReLU(inplace=False))

def conv3x3bnrelu(innum, outnum, stride, pad=1, isre=True):
    if isre:
        return nn.Sequential(nn.Conv2d(innum, outnum, 3, stride, pad),
                      bnrelu(outnum))
    else:
        return nn.Sequential(nn.Conv2d(innum, outnum, 3, stride, pad),
                      nn.BatchNorm2d(outnum))

def convBnRe(innum, outnum, ks, stride, pad=1, isre=True):
    if isre:
        return nn.Sequential(nn.Conv2d(innum, outnum, ks, stride, pad),
                      bnrelu(outnum))
    else:
        return nn.Sequential(nn.Conv2d(innum, outnum, ks, stride, pad),
                      nn.BatchNorm2d(outnum))

def convGnRe(innum, outnum, ks, stride, pad=1, isre=True):
    if isre:
        return nn.Sequential(nn.Conv2d(innum, outnum, ks, stride, pad),
                      bnrelu(outnum))
    else:
        return nn.Sequential(nn.Conv2d(innum, outnum, ks, stride, pad),
                      nn.GroupNorm(16, outnum))


def conv1x1bnrelu(innum, outnum,  stride, isre=True):
    if isre:
        return nn.Sequential(nn.Conv2d(innum, outnum, 1, stride),
                      bnrelu(outnum))
    else:
        return nn.Sequential(nn.Conv2d(innum, outnum, 1, stride),
                      nn.BatchNorm2d(outnum))

def upsample(innum, outnum):
    mindnum = outnum // 2
    return nn.Sequential(nn.Sequential(nn.Conv2d(innum, mindnum, 1, 1)),
                  nn.ConvTranspose2d(mindnum, mindnum, 2, 2),
                         nn.Conv2d(mindnum, outnum, 3, 1, 1))

class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        self.sz = sz

    def forward(self, x):
        avg_pool2d = nn.AvgPool2d(kernel_size=(self.sz[0], self.sz[1]), ceil_mode=False)
        return avg_pool2d(x)

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16, se=None):
        super(SE_Block, self).__init__()
        self.se = se
        self.avg_pool = AdaptiveAvgPool2d(se)                # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        self.weight = self.fc(y).view(b, c, 1, 1)
        weight_list = []
        for i in range(self.se[0]):
            weight_list.append(self.weight)
        self.weight = torch.cat(weight_list, 2)
        weight_list = []
        for i in range(self.se[1]):
            weight_list.append(self.weight)
        self.weight = torch.cat(weight_list, 3)
        res = x * self.weight
        return res





class res(nn.Module):
    ''' Hourglass residual block '''

    def __init__(self, inplanes, outplanes, stride=1, isbn=True):
        super().__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        midplanes = outplanes // 2
        self.bn0 = bnrelu(inplanes)
        self.conv1 = conv1x1bnrelu(inplanes, midplanes, stride)  # bias=False
        self.conv2 = conv3x3bnrelu(midplanes, midplanes, stride)
        self.conv3 = conv1x1bnrelu(midplanes, outplanes, stride)  # bias=False
        if inplanes != outplanes:
            self.conv_skip = nn.Conv2d(inplanes, outplanes, 1, 1)
        self.isbn = isbn

    def forward(self, x):
        residual = x
        if self.isbn:
            out = self.bn0(x)
            out = self.conv1(out)
        else:
            out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.inplanes != self.outplanes:
            residual = self.conv_skip(residual)
        out += residual
        return out


class rese(nn.Module):
    ''' Hourglass residual block '''

    def __init__(self, inplanes, outplanes, stride=1, isbn=True, se=None):
        super().__init__()
        self.stride = stride
        self.inplanes = inplanes
        self.outplanes = outplanes
        midplanes = outplanes // 2
        self.bn0 = bnrelu(inplanes)
        self.conv1 = conv1x1bnrelu(inplanes, midplanes, stride)  # bias=False
        self.conv2 = conv3x3bnrelu(midplanes, midplanes, 1)
        self.conv3 = conv1x1bnrelu(midplanes, outplanes, 1)  # bias=False
        if inplanes != outplanes or stride != 1:
            self.conv_skip = nn.Conv2d(inplanes, outplanes, 1, stride)
        self.isbn = isbn
        self.se = SE_Block(outplanes, se=se)

    def forward(self, x):
        residual = x
        if self.isbn:
            out = self.bn0(x)
            out = self.conv1(out)
        else:
            out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.inplanes != self.outplanes or self.stride != 1:
            residual = self.conv_skip(residual)
        out = self.se(out)
        out += residual

        return out


class hourglass(nn.Module):
    def __init__(self, inplanes, isSe=False, se=None):
        super().__init__()
        if not isinstance(se, type(None)):
            h, w = se
        else:
            h = 24
            w = 24
        se1 = [h, w]
        se2 = [se1[0] // 2, se1[1] // 2]
        se3 = [se2[0] // 2, se2[1] // 2]
        self.conv0 = nn.Conv2d(inplanes, 64, 1, 1)  # bias=False
        self.conv1 = conv3x3bnrelu(inplanes, 32, 2, 1)  # bias=False
        self.conv2 = conv3x3bnrelu(32, 64, 2, 1)
        self.conv3 = conv3x3bnrelu(64, 128, 2, 1)  # bias=False
        self.conv4 = conv3x3bnrelu(128, 128, 1, 1)  # bias=False
        self.res1 = rese(64, 64, se=se3) if isSe else res(64, 64)
        self.res2 = rese(32, 32, se=se2) if isSe else res(32, 32)
        self.res3 = rese(64, 64, se=se1) if isSe else res(64, 64)
        self.upsample1 = upsample(128, 64)
        self.upsample2 = upsample(64, 32)
        self.upsample3 = upsample(32, 64)
        self.bnlu = bnrelu(64)

    def forward(self, x):
        out1 = self.conv0(x)
        out2 = self.conv1(x)
        out3 = self.conv2(out2)
        out = self.conv3(out3)
        out = self.conv4(out)
        out = self.upsample1(out) + out3
        out = self.res1(out)
        out = self.upsample2(out) + out2
        out = self.res2(out)
        out = self.upsample3(out) + out1
        out = self.res3(out)
        return self.bnlu(out)

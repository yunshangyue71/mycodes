import torch
import numpy as np
import cv2

#产生一张图片的anchor box
#step1： 产生baseAnchorsBox， 就是每一个anchor点都会有这几个形状的anchorBox，只是位置不一样而已
#step2：根据特征图的尺寸，产生所有的anchorBox，
#step3：根据特征图上的anchorBox，还没有映射到输入图片上的anchorBox
class AnchorGenerator(object):
    def __init__(self,
                 wsizes,     #[10, 15, 20]一共几种 baseanchor box wide分别是多少
                 hsizes,     #[10, 20, 30] , wratios = [1, 1/2, 1/3, 1/10, 1/5] to make 5 kinds of boxes, 这两个应该相等
                 device = 'cuda',   # m默认， 是cuda数据
                ):
        assert len(wsizes) == len(hsizes), 'len wsizes and len hsizes should same.'
        self.wsizes = torch.Tensor(wsizes).to(device)
        self.hsizes = torch.Tensor(hsizes).to(device)

        self.device = device
        self.baseAnchors = self.__genBaseAnchors()#[n, 4]n表示总共有多少种baseanchor box
    """
    def __genBaseAnchors(self)
        产生baseAnchorsBox， 就是每一个anchor点都会有这几个形状的anchorBox，只是位置不一样而已
    def gridAnchors(self, featmapSize, stride=1, device='cuda')
        根据特征图的尺寸，产生所有的anchorBox， 
    def valid_flags(self, featmap_size, valid_size, device='cuda')
        根据valid
    """
    #return:tensor, (n, 4), n：总共多少总baseAnchorBoxes，4：[x1,y1,x,2,y2]
    def __genBaseAnchors(self):
        baseAnchors = torch.stack([ - 0.5 * (self.wsizes-1),
                                    - 0.5 * (self.hsizes-1),
                                     0.5 * (self.wsizes-1),
                                     0.5 * (self.hsizes-1)],
                                    dim=-1)#.floor()
        return baseAnchors

    #featMapSize:(h,w)
    #stride : img/featmap
    #device：str，  ‘cuda’ 或者‘cpu’
    #return:tensor， [featw, feath, baseBoxNum, 4]
    def genAnchorBoxes(self, featmapSizehw, stride=170):
        feat_h, feat_w = featmapSizehw
        shift_x = torch.arange(0, feat_w, device=self.device).repeat(feat_h).reshape(feat_h, feat_w).to(self.device)
        shift_y = torch.arange(0, feat_h, device=self.device).repeat(feat_w).reshape(feat_w, feat_h)
        shift_y = shift_y.t().to(self.device)
        # shift_y = torch.t(shift_x)

        allAnchors = torch.clone(self.baseAnchors.to(self.device))
        alla = torch.zeros((feat_h,feat_w, len(self.wsizes), 4)).to(self.device)
        alla[:, :, :, 0::2] = allAnchors[None, None, :, 0::2] + shift_x[:, :, None, None] * stride + stride / 2
        alla[:, :, :, 1::2] = allAnchors[None, None, :, 1::2] + shift_y[:, :, None, None] * stride + stride / 2

        return alla

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    hsizes = [4, 5]
    wsizes = [8, 5]
    anchor = AnchorGenerator(
                 wsizes,
                hsizes,
                 )

    a = anchor.baseAnchors
    b = anchor.genAnchorBoxes(featmapSize=(10, 10),stride = 8)
    print('Done')


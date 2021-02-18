import torch
import numpy as np
from utils.bbox_overlaps import bbox_overlaps

class Assigner(object):
    # 经过多次筛选， 每次筛选的结果都是self.anchorGtMat:shape([bboxesPredNum, bboxesGtNum])的真假表
    # 这个只处理 一个featuremap ， 多个level的feature map 请调用多次
    #只是针对一张照片的，
    #输出：
        #anchorBoxLabel:  -1：没有对应的gt
        #anchorBoxIndex： - 1：没有对应的gt
        #anchorBoxIou:  每个anchor box 对应的gt的iou ，没有对应的gt 就为0
    def __init__(self,
                 topk,               # (float): 每个gt最多会选择多少个anchor box 作为pos
                 bboxesAnchor,       # (Tensor): (n, 4). (x1,y1,x2,y2)bboxes还没有判断为pos neg ，也就是该bbox是否是个gt呢？
                 bboxesGt,           # (Tensor): (k, 4). (x1,y1,x2,y2)bboxesPred和bboxesGt进行比较，判断bboxes是否是pos, 尺寸是对应到featuremap上的
                 labelsGt,           # (Tensor,): shape (k, )，每个gt框的label,-1对应background， 0,1，2,3,4对应类别
                 device = 'cuda'     #默认将数据放到 cuda GPU上面
                 ):                  # 0: neg, 没有和gt匹配上， 也就是background； 正数：pos，表示所匹配的gt的index
        self.topk = topk
        self.bboxesAnchor = bboxesAnchor
        self.bboxesGt = bboxesGt.to(device)
        self.labelsGt  = labelsGt.to(device)
        self.device = device

        #初始化计算一些东西
        self.bboxesGtNum = self.bboxesGt.size(0)
        self.bboxesAnchorNum = self.bboxesAnchor.size(0)
        self.iou = bbox_overlaps(self.bboxesAnchor.float(), self.bboxesGt.float())

        """期望的结果"""
        # 存放anchor box和gt box的对应关系的矩阵, 没经过一种方法filter 就会改变一下矩阵
        self.anchorGtMat = torch.ones([self.bboxesAnchorNum, self.bboxesGtNum], dtype=torch.bool, device=self.device)

    # 预测框的center 必须在gt内
    def _anchorBoxCenterInGtFilter(self):
        x1a = self.bboxesAnchor[:, 0].view(self.bboxesAnchorNum, 1)
        y1a = self.bboxesAnchor[:, 1].view(self.bboxesAnchorNum, 1)
        x2a = self.bboxesAnchor[:, 2].view(self.bboxesAnchorNum, 1)
        y2a = self.bboxesAnchor[:, 3].view(self.bboxesAnchorNum, 1)

        cxbox = (self.bboxesGt[:, 0].view(-1) + self.bboxesGt[:, 2].view(-1))/2
        cybox = (self.bboxesGt[:, 1].view(-1) + self.bboxesGt[:, 3].view(-1))/2
        l_ =   x1a - cxbox        #(self.bboxesAnchorNum, self.bboxesGtNum)
        t_ =   y1a - cybox        #(self.bboxesAnchorNum, self.bboxesGtNum)
        r_ = -(x2a - cxbox)       #(self.bboxesAnchorNum, self.bboxesGtNum)
        b_ = -(y2a - cybox)       #(self.bboxesAnchorNum, self.bboxesGtNum)


        #都是true 的表示anchor box 的中心店在 gt里面
        flagL = l_ < 0.01
        flagT = t_ < 0.01
        flagR = r_ < 0.01
        flagB = b_ < 0.01

        # 更新self.anchorGtMat
        posMat = self.anchorGtMat & flagL & flagT & flagR & flagB # (self.bboxesAnchorNum, self.bboxesGtNum) True, False,
        return posMat
        # 在选中的anchor进行 iou进行筛选

    def _iouFilter(self, posMat, neg=0.3):
        # 将没有选中的置为0
        posIouMat = self.iou * posMat
        negMat = torch.logical_and(torch.tensor(self.iou < neg,dtype=torch.bool).to(self.device),
                                   torch.logical_not(posMat))
        return posIouMat, negMat


    def master(self):
        posMat = self._anchorBoxCenterInGtFilter()
        boxGtIndex = torch.nonzero(posMat)

        cls = self.labelsGt[boxGtIndex[:, 1]]
        box = self.bboxesGt[boxGtIndex[:, 1]]
        iou = self.iou[boxGtIndex[:, 0]]
        # iou = iou[:,boxGtIndex[:, 1]]

        posindex = boxGtIndex[:,0]

        posIouMat = self.iou * posMat
        posIouMat, negMat = self._iouFilter(posMat, 0.3)


        return 0

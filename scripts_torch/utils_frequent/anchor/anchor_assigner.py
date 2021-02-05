import torch
import numpy as np
from utils_frequent.IoUs.bbox_overlaps import bbox_overlaps

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
        self.bboxesGt = bboxesGt
        self.labelsGt  = labelsGt
        self.device = device

        #初始化计算一些东西
        self.bboxesGtNum = self.bboxesGt.size(0)
        self.bboxesAnchorNum = self.bboxesAnchor.size(0)

        self.cxGt = (self.bboxesGt[:, 0] + self.bboxesGt[:, 2]) / 2.0  # (x1+x2)/2
        self.cyGt = (self.bboxesGt[:, 1] + self.bboxesGt[:, 3]) / 2.0  # (y1+y2)/2
        self.pointsGt = torch.stack((self.cxGt, self.cyGt), dim=1)

        self.cxAnchor = (self.bboxesAnchor[:, 0] + self.bboxesAnchor[:, 2]) / 2.0  # (x1+x2)/2
        self.cyAnchor = (self.bboxesAnchor[:, 1] + self.bboxesAnchor[:, 3]) / 2.0  # (y1+y2)/2
        self.pointsAnchor = torch.stack((self.cxAnchor, self.cyAnchor), dim=1)

        # (self.bboxesAnchorNum, self.bboxesGtNum), 始终没有改变
        self.iou = bbox_overlaps(self.bboxesAnchor.float(), self.bboxesGt.float())

        """期望的结果"""
        # 存放anchor box和gt box的对应关系的矩阵, 没经过一种方法filter 就会改变一下矩阵
        self.anchorGtMat = torch.ones([self.bboxesAnchorNum, self.bboxesGtNum], dtype=torch.bool, device=self.device)

    #根据anchor box 和gt box 中心点的距离，将距离gt 远的 置为false
    def _centerDistFilter(self):
        # shape （bboxesAnchorNum， bboxesGtNum）
        distances = (self.pointsAnchor[:, None, :] - self.pointsGt[None, :, :]).pow(2).sum(-1).sqrt()

        #将没有选中的置为0
        distances = distances * self.anchorGtMat

        #选择前topk个
        selectable_k = min(self.topk, self.bboxesAnchorNum)  #
        _, topkIndexs = distances.topk(selectable_k, dim=0, largest=False)  # topkIndexs： （topk， bboxesGtNum）

        #过滤对应关系,# 更新self.anchorGtMat
        anchorGtMat = torch.zeros([self.bboxesAnchorNum, self.bboxesGtNum], dtype=torch.bool, device=self.device)
        for i in range(self.bboxesGtNum):
            rows = topkIndexs[:, i]#.view(-1)
            #cols = torch.tensor([i for i in range(self.bboxesGtNum)], device = self.device).repeat(selectable_k)
            anchorGtMat[rows, i] = True
        self.anchorGtMat &= anchorGtMat

    # 在选中的anchor进行 iou进行筛选
    def _iouFilter(self):
        # 将没有选中的置为0
        iou = self.iou * self.anchorGtMat

        #每个gt对应的anchor boxes 的iou ，出去不相交的，后计算平均值
        iouMeanPerBoxGt = iou.sum(0)/(iou>0.0).sum(0)#shape(boxesGt)

        #平均值再加上一个标准差， 作为每个boxGt的阈值， 这要求就会更高了
        # for i in range(iou.size()[1]):
        #     std = iou[:, i].std()
        #     iouMeanPerBoxGt[i] += std

        thredPerBoxGt = iouMeanPerBoxGt

        # 更新self.anchorGtMat
        self.anchorGtMat = iou >= thredPerBoxGt[None, :]  # (self.bboxesAnchorNum, self.bboxesGtNum) True, False,

    # 预测框的center 必须在gt内
    def _anchorBoxCenterInGtFilter(self):
        cxAnchor = self.cxAnchor.view(self.bboxesAnchorNum, 1)
        cyAnchor = self.cyAnchor.view(self.bboxesAnchorNum, 1)

        l_ =   cxAnchor - self.bboxesGt[:, 0].view(-1)        #(self.bboxesAnchorNum, self.bboxesGtNum)
        t_ =   cyAnchor - self.bboxesGt[:, 1].view(-1)        #(self.bboxesAnchorNum, self.bboxesGtNum)
        r_ = -(cxAnchor - self.bboxesGt[:, 2].view(-1))       #(self.bboxesAnchorNum, self.bboxesGtNum)
        b_ = -(cyAnchor - self.bboxesGt[:, 3].view(-1))       #(self.bboxesAnchorNum, self.bboxesGtNum)

        #都是true 的表示anchor box 的中心店在 gt里面
        flagL = l_ > 0.01
        flagT = t_ > 0.01
        flagR = r_ > 0.01
        flagB = b_ > 0.01

        # 更新self.anchorGtMat
        self.anchorGtMat &= flagL & flagT & flagR & flagB # (self.bboxesAnchorNum, self.bboxesGtNum) True, False,

    #一个anchor box 最多对应一个gt box，这一步一定要最后做, 否则就会将太多的bbox 舍去了
    def _anchorBoxAssignOneGtBoxMaxFilter(self):
        #这里使用IoU来进行筛选的
        iou = self.iou * self.anchorGtMat
        index = torch.argmax(iou, dim=1)
        cols = index
        rows = torch.tensor([i for i in range(self.bboxesAnchorNum)]).to(self.device)
        # 更新self.anchorGtMat
        anchorGtMat = torch.zeros([self.bboxesAnchorNum, self.bboxesGtNum], dtype=torch.bool, device= self.device)
        anchorGtMat[rows, cols] = True
        self.anchorGtMat &= anchorGtMat

    # 每一个anchor box对应哪个类别，
    def _output(self):
        #每个anchor box 对应哪个gt,
        #如果没有anchor box 没有对应的gt，下面这个也会max出一个来
        anchorGtMat = self.anchorGtMat.type(torch.int8)
        anchorBoxIndexGtBox = torch.argmax(anchorGtMat, dim=1) #anchor box对应gt bbox的索引
        anchorBoxGtClass = self.labelsGt[anchorBoxIndexGtBox] #anchor box对应gt bbox的标签

        #解决上面的问题， 去掉没有对应上的anchor box
        indexFliter = self.anchorGtMat.any(dim=1)#.bool()
        posIndex = torch.nonzero(indexFliter).reshape(-1)
        indexFliter = torch.bitwise_not(indexFliter)

        #进行过滤
        anchorBoxGtClass[indexFliter] = -1
        anchorBoxIndexGtBox[indexFliter] = -1
        anchorBoxGt = torch.zeros_like(self.bboxesAnchor)
        anchorBoxGt[posIndex] = self.bboxesGt[anchorBoxIndexGtBox[posIndex]]

        infos = {'anchorBoxGt':anchorBoxGt,
                'anchorBoxGtClass':anchorBoxGtClass,
                'anchorBoxIndexGtBox':anchorBoxIndexGtBox,
                'posIndex':posIndex}
        return infos

    def master(self):
        #要进行那些操作，进行assign
        self._centerDistFilter()
        self._iouFilter()
        self._anchorBoxCenterInGtFilter()
        self._anchorBoxAssignOneGtBoxMaxFilter()

        return self._output()

if __name__ == '__main__':
    #产生anchorbox
    from anchorbox_generate import AnchorGenerator
    baseSize = 10
    scales = [1]
    hratios = [1]
    featuremapSize = [9, 9]
    anchorBoxes = AnchorGenerator(
        baseSize,  # size length basic
        scales,    # [1., 10., 15.,] to make 3 kinds of boxes, w and h * scales
        hratios,   # [1,2,3,10.5] , wratios = [1, 1/2, 1/3, 1/10, 1/5] to make 5 kinds of boxes
        scaleMajor = True,  # when times , first multiply scale
    ).gridAnchors(featmapSize=featuremapSize)

    #得到gtbox
    gtBox = [[1, 1, 3, 3],
         [5, 5, 8, 8],
             [2,2,7,7]]
    gtBox = np.array(gtBox)
    gtBox = torch.from_numpy(gtBox)
    device = torch.device('cuda:0')
    gtBox = gtBox.to(device)

    #得到gtlabel
    labelGt = [1, 4, 5]
    labelGt = np.array(labelGt)
    labelGt = torch.from_numpy(labelGt)

    assign = Assigner(9, anchorBoxes, gtBox, labelGt)
    anchorBoxGtClass, anchorBoxIndexGtBox = assign.master()

    #画图显示出来,这里自己话的时候理解错了， 讲bbox的尺寸花在了feature map上面了，其实还是应该经过缩放的
    def plotBox(box,c = 'r', w= 1):
        x1, y1, x2, y2 = box
        x = [x1, x2, x2, x1, x1]
        y = [y1, y1, y2, y2, y1]
        plt.plot(x, y, c=c, linewidth= w)

    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    rows = np.array([i for i in range(featuremapSize[0])])
    cols = np.array([i for i in range(featuremapSize[1])])
    xx, yy = np.meshgrid(rows, cols)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    plt.scatter(xx, yy, c='k')

    box = gtBox[0]
    for i in range(len(gtBox)):
        box = gtBox[i]
        plotBox(box, w = 3)

    anchorBoxes = anchorBoxes.to('cpu').numpy()
    for i in range(anchorBoxes.shape[0]):
        if anchorBoxIndexGtBox[i] != -1:
            plotBox(anchorBoxes[i], c = 'b')
    plt.show()
    print(assign)
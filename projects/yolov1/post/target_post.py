import torch
from post.anchorbox_generate import AnchorGenerator
from post.anchor_assigner import Assigner

class Post(object):
    def __init__(self, wsizes, hsizes, featuremapSize, stride):

        self.anchorBoxes = AnchorGenerator(
            wsizes,  # [10, 15, 20]一共几种 baseanchor box wide分别是多少
            hsizes,  # [10, 20, 30] , wratios = [1, 1/2, 1/3, 1/10, 1/5] to make 5 kinds of boxes, 这两个应该相等
            device='cuda',  # m默认， 是cuda数据
        ).genAnchorBoxes(featmapSizehw=featuremapSize, stride = stride)


    def forward(self, boxGt, clsGt):
        for i in range(boxGt.size()[0]):
            boxGti = boxGt[i]
            boxGti[:, 2:] += boxGti[:, :2]
            clsGti = clsGt[i]
            boxGtAnchor = Assigner(
                9,
                self.anchorBoxes.reshape(-1, 4),  # (Tensor): (n, 4). (x1,y1,x2,y2)bboxes还没有判断为pos neg ，也就是该bbox是否是个gt呢？
                boxGti.reshape(-1, 4),  # (Tensor): (k, 4). (x1,y1,x2,y2)bboxesPred和bboxesGt进行比较，判断bboxes是否是pos, 尺寸是对应到featuremap上的
                clsGti.reshape(-1),  # (Tensor,): shape (k, )，每个gt框的label,-1对应background， 0,1，2,3,4对应类别
                device = 'cuda'  # 默认将数据放到 cuda GPU上面
            ).master()

        return boxGtAnchor

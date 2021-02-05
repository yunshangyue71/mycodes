import torch
from torch import nn
import torch.nn.functional as F

# boxes分布的函数
# <https://arxiv.org/abs/2006.04388>`_.
# 一共feath*featw个anchor， 每个anchor 点回归了n个box， 每个box 用两个点(x1,y1,x2,y2)或者4个值(dist_up,dist_down,dist_left,dist_right)
#pred(featw*feath*4, n)
#假如说某个anchor的distUp, label是4.8，pred = 4.3；
#那么pred 就和4做交叉熵损失，然后系数是0.2； 加上 pred和5做交叉熵， 然后系数是0.8， 因为label和5近，所以系数就大
class DistributionFocalLoss(object):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def cal(self,
            pred,   #typee:(torch.Tensor);shape(N, n), n:一个anchor point 最终回归了多少个box，预测的距离是多少个anchor， 也就是除以stride后的
            # N， 所有的anchor point 对应4个距离，上下左右， N= anchorPoitNum*4 ; Predicted general distribution of bounding boxes
            label,  #label:(torch.Tensor);shape (N,);: Target distance label for bounding boxes
            ): #Returns: torch.Tensor: Loss tensor with shape (N,).
        #label  到左边点的距离
        dis_left = label.floor().long()
        weight_right = label - dis_left.float()

        #label 到右边点的距离
        dis_right = dis_left + 1
        weight_left = dis_right.float() - label

        #loss
        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left \
               + F.cross_entropy(pred, dis_right, reduction='none') * weight_right
        return loss
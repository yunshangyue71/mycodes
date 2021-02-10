import torch
from torch import nn
import torch.nn.functional as F

#综合评估 box iou质量、分类质量的损失函数
#<https://arxiv.org/abs/2006.04388>
#最终的loss shape（N，C）， 每个anchor 点预测的score 表示分类和iou 质量的综合情况
#如果某个anchor 没有对应的gt， 那么这个anchor point的综合得分就是0， loss：pred和0进行交叉熵，系数是pred的beta倍
#如果某个anchor 有对应的gt， 那么这个anchor point 的综合得分就是score， loss：pred 和score的交叉熵， 系数是（score - pred） 的beta倍
class QualityFocalLoss(object):
    def __init__(self,
                 use_sigmoid=True,  #typee: (bool);  Defaults to True. 是否用sigmoid进行损失计算，默认是
                 beta=2.0,          #typee:float; Defaults to 2.0.
                ):
        super(QualityFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta

    def cal(self,
           pred, #typee:torch.Tensor;shape:(N,C), C class Number;val：这个anchor iou以及cls综合起来代表各个类别的能力
           target,#（label， score); type:torch.tensor;shape:(N,);means:对应C中的哪个类别 | type:torch.tensor; shape:(N, )；means：iou score
           beta=2.0):
        assert len(target) == 2, """target for QFL must be a tuple of two elements, including category label and quality label, respectively"""
        label, score = target

        """loss初始化"""
        #pred 每个值，进行sigmoid ->(0, 1); (N, C)
        pred_sigmoid = pred.sigmoid()
        scale_factor = pred_sigmoid
        zerolabel = scale_factor.new_zeros(pred.shape)
        #loss的初始化， 每个值和0进行二值交叉熵， 0表示根本不可能预测出gt
        #(1-t)log(p)*p^2, 交叉熵多一个p2
        loss = F.binary_cross_entropy_with_logits(
            pred, zerolabel, reduction='none') * scale_factor.pow(beta)
        """END"""


        """loss 赋值"""
        #每个pred(N,C), 每个anchor 最多可以对应一个类别
        #某个anchor 对应某个类别， 那就将这个值进行更改

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = pred.size(1)
        #找到pos的index
        pos = torch.nonzero(label >= 0, as_tuple=False).squeeze(1)
        #找到pos的label
        pos_label = label[pos].long()
        scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
        loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
            pred[pos, pos_label], score[pos],
            reduction='none') * scale_factor.abs().pow(beta)

        loss = loss.sum(dim=1, keepdim=False)
        return loss
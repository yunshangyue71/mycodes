import math
import torch
import torch.nn as nn
from utils_frequent.IoUs.multi_iou_cal import MultiIoUCal

class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps                  #Eps to avoid log(0).

    def forward(self,
                bboxesPred,  # bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
                bboxesGt,  # bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
                ):
        ious = MultiIoUCal(bboxesPred, bboxesGt, isAligned = True).clamp(min=self.eps)
        loss = -ious.log()
        return loss

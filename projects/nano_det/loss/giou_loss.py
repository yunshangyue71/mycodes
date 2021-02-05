import math
import torch
import torch.nn as nn
from loss.multi_iou_cal import MultiIoUCal

class GIoULoss(nn.Module):

    def __init__(self, eps=1e-6):
        super(GIoULoss, self).__init__()
        self.eps = eps
    def forward(self,
                bboxesPred,  # bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
                bboxesGt,  # bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
                ):
        gious = MultiIoUCal(bboxesPred, bboxesGt, mode='giou', isAligned = True).iouResult().clamp(min=self.eps)
        loss = 1 - gious
        return loss


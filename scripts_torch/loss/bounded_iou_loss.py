import math
import torch
import torch.nn as nn
from utils_frequent.IoUs.multi_iou_cal import MultiIoUCal

def bounded_iou_loss(pred, target, beta=0.2, eps=1e-3):
    """BIoULoss.

    This is an implementation of paper
    `Improving Object Localization with Fitness NMS and Bounded IoU Loss.
    <https://arxiv.org/abs/1711.00164>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Target bboxes.
        beta (float): beta parameter in smoothl1.
        eps (float): eps to avoid NaN.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]
    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - torch.max(
        (target_w - 2 * dx.abs()) /
        (target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
    loss_dy = 1 - torch.max(
        (target_h - 2 * dy.abs()) /
        (target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w /
                            (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h /
                            (target_h + eps))
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh],
                            dim=-1).view(loss_dx.size(0), -1)

    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta,
                       loss_comb - 0.5 * beta)
    return loss


class BoundedIoULoss(nn.Module):
    def __init__(self, beta=0.2, eps=1e-3):
        super(BoundedIoULoss, self).__init__()
        self.beta = beta
        self.eps = eps

    def forward(self,
                bboxesPred,  # bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
                bboxesGt,  # bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
                ):

        loss = bounded_iou_loss(
            bboxesPred,  # bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
            bboxesGt,  # bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            beta=self.beta,
            eps=self.eps,
            )
        return loss

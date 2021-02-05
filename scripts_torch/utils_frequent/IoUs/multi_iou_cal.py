import math
import torch
import torch.nn as nn

class MultiIoUCal(object):
    #result: Tensor: shape(m, n) if ``is_aligned `` is False else shape(m, )
    def __init__(self,
         bboxes1,           #bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
         bboxes2,           #bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
         mode = 'iou',      #'iou'; iof: area1/ union;giou: iou - union/outerArea
         isAligned = False, #False: cal each pred bbox to all the gt bbox
         eps = 1e-6         #eps (float, optional): A value added to the denominator for numerical stability
                 ):
        assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
        assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)
        assert bboxes1.shape[:-2] == bboxes2.shape[:-2] # Batch dim must be the same, Batch dim: (B1, B2, ... Bn)
        assert mode in ['iou', 'iof', 'giou']
        self.bboxes1 = bboxes1
        self.bboxes2 = bboxes2
        self.mode = mode
        self.isAligned = isAligned
        self.eps = eps

        self.area1 = (self.bboxes1[..., 2] - self.bboxes1[..., 0]) * \
                     (self.bboxes1[..., 3] - self.bboxes1[..., 1])
        self.area2 = (self.bboxes2[..., 2] - self.bboxes2[..., 0]) * \
                     (self.bboxes2[..., 3] - self.bboxes2[..., 1])
        self.bboxes1Num = bboxes1.size(-2)
        self.bboxes2Num = bboxes2.size(-2)
        if self.isAligned:
            assert self.bboxes1Num == self.bboxes2Num

    def _calAlignedComponents(self):
        self.p1  = torch.max(self.bboxes1[..., :2], self.bboxes2[..., :2])  # [B, rows, 2]
        self.p2  = torch.min(self.bboxes1[..., 2:], self.bboxes2[..., 2:])  # [B, rows, 2]
        self.wh = (self.p2 - self.p1).clamp(min=0)  # [B, rows, 2]
        self.overlap = self.wh[..., 0] * self.wh[..., 1]
        if self.mode in ['iou', 'giou']:
            self.union = self.area1 + self.area2 - self.overlap
        else:
            self.union = self.area1
        if self.mode == 'giou':
            self.enclosed_lt = torch.min(self.bboxes1[..., :2], self.bboxes2[..., :2])
            self.enclosed_rb = torch.max(self.bboxes1[..., 2:], self.bboxes2[..., 2:])

    def _calNotAlignedComponents(self):
        self.p1 = torch.max(self.bboxes1[..., :, None, :2],
                       self.bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        self.p2 = torch.min(self.bboxes1[..., :, None, 2:],
                       self.bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]
        self.wh = (self.p2 - self.p1).clamp(min=0)  # [B, rows, cols, 2]
        self.overlap = self.wh[..., 0] * self.wh[..., 1]
        if self.mode in ['iou', 'giou']:
            self.union = self.area1[..., None] + self.area2[..., None, :] - self.overlap
        if self.mode in ['iof']:
            self.union = self.area1[..., None]
        if self.mode == 'giou':
            self.enclosed_lt = torch.min(self.bboxes1[..., :, None, :2],
                                         self.bboxes2[..., None, :, :2])
            self.enclosed_rb = torch.max(self.bboxes1[..., :, None, 2:],
                                         self.bboxes2[..., None, :, 2:])
    def iouResult(self):
        aPicBoxShapeshape = self.bboxes1.shape[:-2]
        if self.bboxes1Num == 0 or self.bboxes2Num == 0:
            if self.isAligned:
                return self.bboxes1.new(aPicBoxShapeshape + (self.bboxes1Num,))
            else:
                return self.bboxes1.new(aPicBoxShapeshape + (self.bboxes1Num, self.bboxes2Num))

        if self.isAligned:
            self._calAlignedComponents()
        else:
            self._calNotAlignedComponents()

        self.eps = self.union.new_tensor([self.eps])
        self.union = torch.max(self.union, self.eps)
        ious = self.overlap / self.union

        if self.mode in ['iou', 'iof']:
            return ious
        if self.mode in ['giou']:
            # calculate gious
            enclose_wh = (self.enclosed_rb - self.enclosed_lt).clamp(min=0)
            enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
            enclose_area = torch.max(enclose_area, self.eps)
            gious = ious - (enclose_area - self.union) / enclose_area
            return gious
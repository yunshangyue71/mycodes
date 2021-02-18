import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class yoloLoss(nn.Module):
    def __init__(self):
        pass
    # def __init__(self, S, B, l_coord, l_noobj):
    #     super(yoloLoss, self).__init__()
    #     self.S = S
    #     self.B = B
    #     self.l_coord = l_coord
    #     self.l_noobj = l_noobj

    def forward(self, pred, target):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        '''
        objMask = target[:,:,:, 4] == 0
        noobjMask = target[:,:,:,4] > 0

        objMask = objMask.expand_as(target)
        noobjMask = noobjMask.expand_as(target)

        boxPred, clsPred = pred
        boxTarget, clsTarget = target

        lossCls = torch.pow(clsPred - clsTarget, 2).sum()

        lossScore = torch.pow(scorePred - scoreTarget, 2).sum()
        lossScore = torch.pow(scorePred - scoreTarget, 2).sum()

        #lossBox = torch.()

        #loss = +++


        N = pred_tensor.size()[0]
        coo_mask = target_tensor[:, :, :, 4] > 0
        noo_mask = target_tensor[:, :, :, 4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        coo_pred = pred_tensor[coo_mask].view(-1, 30)
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)  # box[x1,y1,w1,h1,c1]
        class_pred = coo_pred[:, 10:]  # [x2,y2,w2,h2,c2]

        coo_target = target_tensor[coo_mask].view(-1, 30)
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        class_target = coo_target[:, 10:]

        # compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1, 30)
        noo_target = target_tensor[noo_mask].view(-1, 30)
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size())
        noo_pred_mask.zero_()
        noo_pred_mask[:, 4] = 1;
        noo_pred_mask[:, 9] = 1
        noo_pred_c = noo_pred[noo_pred_mask]  # noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, size_average=False)

        # compute contain obj loss
        coo_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).cuda()
        for i in range(0, box_target.size()[0], 2):  # choose the best iou box
            box1 = box_pred[i:i + 2]
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:, :2] = box1[:, :2] / 14. - 0.5 * box1[:, 2:4]
            box1_xyxy[:, 2:4] = box1[:, :2] / 14. + 0.5 * box1[:, 2:4]
            box2 = box_target[i].view(-1, 5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:, :2] = box2[:, :2] / 14. - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / 14. + 0.5 * box2[:, 2:4]
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])  # [2,1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()

            coo_response_mask[i + max_index] = 1
            coo_not_response_mask[i + 1 - max_index] = 1

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        box_target_iou = Variable(box_target_iou).cuda()
        # 1.response loss
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        box_target_response = box_target[coo_response_mask].view(-1, 5)
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], size_average=False)
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], size_average=False) + F.mse_loss(
            torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]), size_average=False)
        # 2.not response loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0
        # not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)

        # I believe this bug is simply a typo
        not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], size_average=False)

        # 3.class loss
        class_loss = F.mse_loss(class_pred, class_target, size_average=False)

        return (
                           self.l_coord * loc_loss + 2 * contain_loss + not_contain_loss + self.l_noobj * nooobj_loss + class_loss) / N





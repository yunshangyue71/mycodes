"""
pred box和box 在图片中实际的位置的转换
"""
import torch

#相对于anchor point的距离，转化为相对于图像左上角的距离
#predBoxes：type:(tensor);shape(b, featw, feath, boxNum*4);形式：(x1y1x2y2)
#featSize:type(tuple):shape(featw, feath)
#stride: 8, image/featmap
def anchorPointDist2LUDist(predBoxes, featSize, stride,device='cuda'):
    assert predBoxes.shape[1] == featSize[0] and predBoxes.shape[2] == featSize[1]
    featw, feath = featSize
    predBoxes_ = torch.clone(predBoxes)
    shiftx = torch.arange(0, featw).repeat(feath).reshape(featw, feath).to(device)+0.5
    shifty = torch.t(shiftx)
    predBoxes_[:, :, :, 0::2] = shiftx[None, :, :, None] * stride + stride/2 - predBoxes_[:, :, :, 0::2]
    predBoxes_[:, :, :, 1::2] = shifty[None, :, :, None] * stride + stride / 2-predBoxes_[:, :, :, 1::2]
    predBoxes_[:, :, :, 2::2] = predBoxes_[:, :, :, 2::2] + shiftx[None, :, :, None] * stride + stride /2
    predBoxes_[:, :, :, 3::2] = predBoxes_[:, :, :, 3::2] + shifty[None, :, :, None] * stride + stride / 2
    return predBoxes_

def LUDist2AnchorPointDist(predBoxes, featSize, stride,device='cuda'):
    assert predBoxes.shape[1] == featSize[0] and predBoxes.shape[2] == featSize[1]
    featw, feath = featSize
    predBoxes_ = torch.clone(predBoxes)
    shiftx = torch.arange(0, featw).repeat(feath).reshape(featw, feath).to(device)+0.5
    shifty = torch.t(shiftx)
    predBoxes_[:, :, :, 0::2] = shiftx[None, :, :, None] * stride + stride / 2 - predBoxes_[:, :, :, 0::2]
    predBoxes_[:, :, :, 1::2] = shifty[None, :, :, None] * stride + stride / 2 - predBoxes_[:, :, :, 1::2]
    predBoxes_[:, :, :, 2::2] = predBoxes_[:, :, :, 2::2] + shiftx[None, :, :, None] * stride + stride / 2
    predBoxes_[:, :, :, 3::2] = predBoxes_[:, :, :, 3::2] + shifty[None, :, :, None] * stride + stride / 2
    return predBoxes_

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)
if __name__ == '__main__':
    predBoxes = torch.zeros((1,5,5,8), dtype=torch.float16)
    featSize = (5, 5)
    stride  = 1
    mish = anchorPointDist2LUDist(predBoxes, featSize, stride)#torch.range(0, 9).repeat(10).reshape(10,10)

    print(mish)
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import cv2
from dataload.imgaug_wo_shape import ImgAugWithoutShape
from dataload.imgaug_w_shape import ImgAugWithShape
from dataload.resize_uniform import resizeUniform
"""
一个image一个anno.txt
imageName.txt 
    xmin, ymin, w,h, cls0,cls1
    xmin, ymin, w,h, cls0,cls1
    xmin, ymin, w,h, cls0,cls1
这几个参数都是根目录
output
    img 0-1
"""
class ListDataset(Dataset):
    def __init__(self,
                 trainAnnoPath,  # txt files root /
                 trainImgPath,  # images root /
                 netInputSizehw=(320, 320),
                 augFlag=False,
                 normalize = None
                 ):
        self.trainAnnoPath = trainAnnoPath
        self.trainImgPath = trainImgPath
        self.netInputSizehw = tuple(netInputSizehw)
        self.annNames = os.listdir(self.trainAnnoPath)#[:16]
        self.normalize = np.array(normalize)
        self.augFlag = augFlag
        self.showFlag = 0


        self.maskChannelNumber = 1
        self.imgChannelNumber = 1

    def __getitem__(self, index):
        """bbox img org"""
        maskPath = self.trainAnnoPath + self.annNames[index]
        if self.imgChannelNumber == 3:
            img = cv2.imread(self.trainImgPath + self.annNames[index].split('.')[0] + '.jpg' , cv2.COLOR_BGR2RGB)
        if self.imgChannelNumber == 1:
            img = cv2.imread(self.trainImgPath + self.annNames[index].split('.')[0] + '.jpg', cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32)

        if self.maskChannelNumber == 3:
            mask = cv2.imread(maskPath)# , cv2.COLOR_BGR2RGB)
        if self.maskChannelNumber == 1:
            mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.uint8)

        winName = self.annNames[index]
        if self.showFlag:
            cv2.imshow(winName+"org", img.astype(np.uint8))
            cv2.imshow(winName+"org_mask", mask.astype(np.uint8))

        """unifor resize 放在最后，输入网络的图片会有很多的0， 经过imgaug这些将会变为非0有利于学习"""
        img, infos, _ = resizeUniform(img, self.netInputSizehw)
        mask, infos, _ = resizeUniform(mask, self.netInputSizehw)
        if self.showFlag:
            cv2.imshow(winName+"resize", img.astype(np.uint8))
            cv2.imshow(winName+"resize_mask", mask.astype(np.uint8))
        if self.augFlag:
            """Img Aug With Shape, 放射变换的增强一定要放在前面，主要是0的情况"""
            imgauger = ImgAugWithShape(img, bboxes)
            imgauger.shear(15)
            imgauger.translate(translate=[-0.2, 0.2])
            img, bboxes = (imgauger.img, imgauger.boxes)

            if self.showFlag:
                cv2.imshow(winName + "_augshape", img.astype(np.uint8))
                cv2.imshow(winName + "_augshape", mask.astype(np.uint8))

            """非放射变换，放在最后， 最后的img 不用clip到（0，1）之间"""
            imgauger = ImgAugWithoutShape(img)
            imgauger.brightness()
            imgauger.constrast()
            imgauger.saturation()
            imgauger.normalize1(mean = self.normalize[0], std= self.normalize[1])
            img = imgauger.img
            if self.showFlag:
                cv2.imshow(winName + "_augcolor", img.astype(np.uint8))


            if self.showFlag:
                outwh = (80,80)
                cv2.imshow(winName + "_netout", np.copy(cv2.resize(img,(outwh[0], outwh[1]))).astype(np.uint8))
            if self.showFlag: cv2.waitKey()

        """return 两种return可供选择"""
        if self.imgChannelNumber == 1:
            img = img[:, :, np.newaxis]
        img = img.transpose(2, 0, 1)  # 因为pytorch的格式是CHW
        if self.maskChannelNumber == 1:
            mask = mask[:, :, np.newaxis]
        mask = mask.transpose(2, 0, 1)
        meta = dict(images=torch.from_numpy(img.astype(np.float32)),
                    masks = torch.from_numpy(mask.astype(np.uint8)),
                    annoName = self.annNames[index])
        return meta

    def __len__(self):
        return len(self.annNames)


if __name__ == '__main__':
    pass
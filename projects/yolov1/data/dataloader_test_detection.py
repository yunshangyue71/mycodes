import numpy as np
import os
import torch
from torch.utils.data import Dataset
import cv2
from data.imgaug_wo_shape import ImgAugWithoutShape
from data.imgaug_w_shape import ImgAugWithShape
from data.resize_uniform import resizeUniform
from VOCdataset import vocAnnoPathes, parseVoc
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
                 testImgPath,  # images root /
                 netInputSizehw,
                 imgChannelNumber,
                 ):
        self.testImgPath = testImgPath
        self.netInputSizehw = tuple(netInputSizehw)
        self.imgChannelNumber = imgChannelNumber

        self.imgNames = os.listdir(testImgPath)
        self.showFlag = 0

    def __getitem__(self, index):
        """bbox img org"""

        """img channel number"""
        if self.imgChannelNumber == 3:
            img = cv2.imread(self.testImgPath  + self.imgNames[index].split('.')[0] + '.jpg')# cv2.COLOR_BGR2RGB)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.imgChannelNumber == 1:
            img = cv2.imread(self.testImgPath  + self.imgNames[index].split('.')[0] + '.jpg', cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32)

        winName = ""#self.annNames[index]
        # if self.showFlag:
        #     cv2.imshow(winName + "origin", img)

        """unifor resize 放在最后，输入网络的图片会有很多的0， 经过imgaug这些将会变为非0有利于学习"""
        img, infos, bboxes = resizeUniform(img, self.netInputSizehw)
        if self.showFlag:
            cv2.imshow(winName + "resize", img)
        if self.showFlag:
            outwh = (7,7)
            cv2.imshow(winName + "out", img)
        if self.showFlag:
            print(self.imgNames[index])
            cv2.waitKey()
            #cv2.destroyAllWindows()

        if self.imgChannelNumber == 1:
            img = img[:, :, np.newaxis]

        """return 两种return可供选择"""
        img = img.transpose(2, 0, 1)  # 因为pytorch的格式是CHW
        meta = dict(images = torch.from_numpy(img.astype(np.float32)),
                    annoName = self.imgNames[index])

        #"""如果每个img输出的形状一样， 那么就可以下面"""
        # return torch.from_numpy(img.astype(np.float32)), \
        #        torch.from_numpy(bboxes.astype(np.float32)),\
        #        torch.from_numpy(classes.astype(np.float32))
        # 每张图片的bbox数目不一致，因此要使用这样使用, 在使用数目不一致的元素，就不能batch是使用，只能一个照片一张照片
        # torch.utils_less.data.DataLoader(trainData, collate_fn=collate_function,)这个需要collate_fn就需要制定一下了
        return meta

    def __len__(self):
        return len(self.imgNames)


if __name__ == '__main__':
    pass
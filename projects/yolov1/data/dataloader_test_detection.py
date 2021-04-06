import numpy as np
import os
import torch
from torch.utils.data import Dataset
import cv2
from data.resize_uniform import resizeUniform
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
                 imgPath,  # images root /
                 netInputSizehw,
                 imgChannelNumber,
                 clsname = {0: "person"}
                 ):
        self.imgPath = imgPath
        self.netInputSizehw = tuple(netInputSizehw)
        self.imgNames = os.listdir(self.imgPath)[:100] # format me
        self.imgChannelNumber = imgChannelNumber
        self.clsname = clsname
        self.showFlag = 1

    def __getitem__(self, index):
        """input img rgb or gray"""
        if self.imgChannelNumber == 3:
            img = cv2.imread(self.imgPath + self.imgNames[index].split('.')[0] + '.jpg')# cv2.COLOR_BGR2RGB)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.imgChannelNumber == 1:
            img = cv2.imread(self.imgPath + self.imgNames[index].split('.')[0] + '.jpg', cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32)

        """unifor resize 放在最后，输入网络的图片会有很多的0， 经过imgaug这些将会变为非0有利于学习"""
        img, infos = resizeUniform(img, self.netInputSizehw)

        if self.showFlag:
            # winName = self.imgNames[index]
            winName = ""
            cv2.imshow(winName, img)
            print(self.imgNames[index])
            cv2.waitKey()
            #cv2.destroyAllWindows()

        # make dim == 3
        if self.imgChannelNumber == 1:
            img = img[:, :, np.newaxis]

        """return 两种return可供选择"""
        img = img.transpose(2, 0, 1)  # 因为pytorch的格式是CHW
        meta = dict(images=torch.from_numpy(img.astype(np.float32)),
                    imgName = self.imgNames[index])
        return meta

    def __len__(self):
        return len(self.imgNames)
if __name__ == '__main__':
    pass
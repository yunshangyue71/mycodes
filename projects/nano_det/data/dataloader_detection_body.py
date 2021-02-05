import numpy as np
import os
import torch
from torch.utils.data import Dataset
import cv2
from data.imgaug_wo_shape import ImgAugWithoutShape
from data.imgaug_w_shape import ImgAugWithShape
from utils.resize_uniform import resizeUniform
"""
一个image一个anno.txt
imageName.txt 
    xmin, ymin, x2,y2, cls0,cls1  wider person
    xmin, ymin, w,h, cls0,cls1
    xmin, ymin, w,h, cls0,cls1
这几个参数都是根目录
output
    img 0-1

背景：-1
"""


class ListDataset(Dataset):
    def __init__(self,
                 trainAnnoPath,  # txt files root /
                 trainImgPath,  # images root /
                 netInputSizehw=(320, 320),
                 augFlag=False,
                 ):
        self.trainAnnoPath = trainAnnoPath
        self.trainImgPath = trainImgPath
        self.netInputSizehw = tuple(netInputSizehw)
        self.annNames = os.listdir(self.trainAnnoPath)
        self.augFlag = augFlag

    def __getitem__(self, index):
        """bbox"""
        txtPath = self.trainAnnoPath + self.annNames[index]
        infos = np.loadtxt(txtPath, skiprows=1)
        infos = np.array(infos, dtype=np.float32).reshape(-1, 5)

        bboxes = infos[:, 1:]  # .reshape(-1, 5)
        bboxes[:, 2:] -= bboxes[:, :2]
        classes = np.array(infos[:, 0] > 2 , dtype= np.float)

        #print(classes.shape)

        """img """
        img = cv2.imread(self.trainImgPath + self.annNames[index].split('.')[0] + '.jpg' , cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255

        """aug"""
        if self.augFlag:
            """aug images"""
            imgAug = np.copy(img)
            imgAug = self.imgAuger(imgAug)  # 这里都是0-1的增强， 所以要注意
            imgAug = np.clip(imgAug, 0, 1)
            '''END'''

            """aug images and boxes"""
            bboxesAug_ = np.copy(bboxes)
            # (x1,y1, w,h)->(x1,y1, x2,y2)
            bboxesAug_[:, 2:] = bboxesAug_[:, :2] + bboxesAug_[:, 2:]
            imgAug, bboxesAug = self.imgBoxAuger(imgAug, bboxesAug_)

            # (x1,y1, x2,y2)->(x1,y1, w,h)
            bboxesAug[:, 2:] = bboxesAug[:, 2:] - bboxesAug[:, :2]
            # bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
            """END"""
        else:
            # print('aa')
            imgAug = np.copy(img)
            bboxesAug = np.copy(bboxes)

        """resize to input size"""
        #imgAug = cv2.resize(imgAug, self.netInputSizehw[::-1])
        imgAug, effectArea, realh, realw = resizeUniform(imgAug, self.netInputSizehw)

        hsRate = realh / img.shape[0]
        wsRate = realw / img.shape[1]
        bboxesAug[:, 0:4:2] *= wsRate
        bboxesAug[:, 0] += effectArea['x']
        bboxesAug[:, 1:4:2] *= hsRate
        bboxesAug[:, 1] += effectArea['y']

        """return"""
        imgout = imgAug.transpose(2, 0, 1)  # 因为pytorch的格式是CHW
        meta = dict(images=torch.from_numpy(imgout.astype(np.float32)),
                    bboxesGt=bboxesAug,
                    classes=classes,
                    imgname = self.annNames[index].split('.')[0] + '.jpg')

        '''show'''
        showFlag = 1
        if showFlag:
            """需要设置的"""
            imgOutSize = (60, 40)  # 特征图最后的输出大小

            """END"""
            color = (0, 0, 255)
            thick = 1
            for i in range(bboxes.shape[0]):
                cv2.rectangle(img, tuple((int(bboxes[i][0]), int(bboxes[i][1]))),
                              tuple((int(bboxes[i][0]) + int(bboxes[i][2]),
                                     int(bboxes[i][1]) + int(bboxes[i][3]))),
                              color, thick)
            cv2.imshow(self.annNames[index] + '_org', img)

            for i in range(bboxesAug.shape[0]):
                cv2.rectangle(imgAug, tuple((int(bboxesAug[i][0]), int(bboxesAug[i][1]))),
                              tuple((int(bboxesAug[i][0]) + int(bboxesAug[i][2]),
                                     int(bboxesAug[i][1]) + int(bboxesAug[i][3]))),
                              color, thick)
            cv2.imshow(self.annNames[index] + '_aug', imgAug)

            imgOut = cv2.resize(imgAug, imgOutSize)
            hsRate2 = imgOutSize[0] / imgAug.shape[0]
            wsRate2 = imgOutSize[1] / imgAug.shape[1]
            bboxesOut = np.copy(bboxesAug)
            bboxesOut[:, 0:4:2] *= wsRate2
            bboxesOut[:, 1:4:2] *= hsRate2
            for i in range(bboxes.shape[0]):
                cv2.rectangle(imgOut, tuple((int(bboxesOut[i][0]), int(bboxesOut[i][1]))),
                              tuple((int(bboxesOut[i][0]) + int(bboxesOut[i][2]),
                                     int(bboxesOut[i][1]) + int(bboxesOut[i][3]))),
                              color, thick)
                print(classes[i])
            cv2.imshow(self.annNames[index] + '_out', imgOut)
            cv2.waitKey()
            cv2.destroyAllWindows()

        """如果每个img输出的形状一样， 那么就可以下面"""
        # return torch.from_numpy(imgout.astype(np.float32)), \
        #        torch.from_numpy(bboxes.astype(np.float32)),\
        #        torch.from_numpy(classes.astype(np.float32))

        # 每张图片的bbox数目不一致，因此要使用这样使用, 在使用数目不一致的元素，就不能batch是使用，只能一个照片一张照片
        # torch.utils_less.data.DataLoader(trainData, collate_fn=collate_function,)这个需要collate_fn就需要制定一下了
        return meta

    def __len__(self):
        return len(self.annNames)

    #####
    def imgAuger(self, img):
        aug = ImgAugWithoutShape(img)
        aug.brightness(delta=0.2, prob=0.5)
        aug.constrast(alphaLow=0.8, alphaUp=1.2, prob=0.5)
        aug.constrast(alphaLow=0.8, alphaUp=1.2, prob=0.5)
        return aug.img

    def imgBoxAuger(self, img, boxes):
        aug = ImgAugWithShape(img, boxes)
        aug.scale(ratio=(0.8, 1.2), prob=0.5)
        aug.shear(3, prob=0.5)
        aug.stretch(width_ratio=(1, 1), height_ratio=(1, 1), prob=0)
        aug.rotation(degree=20, prob=1)

        return aug.img, aug.boxes


if __name__ == '__main__':
    pass
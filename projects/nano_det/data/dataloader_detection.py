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
        self.showFlag = 1

    def __getitem__(self, index):
        """bbox img org"""
        txtPath = self.trainAnnoPath + self.annNames[index]
        infos = np.loadtxt(txtPath)
        infos = np.array(infos, dtype=np.float32).reshape(-1, 5)

        bboxes = infos[:, :4]
        classes = infos[:, 4:]
        img = cv2.imread(self.trainImgPath + self.annNames[index].split('.')[0] + '.jpg')  # , cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)

        if self.showFlag: self.__show(np.copy(img).astype(np.uint8), bboxes, classes, self.annNames[index], color = (0, 0, 255))

        """unifor resize 放在最后，输入网络的图片会有很多的0， 经过imgaug这些将会变为非0有利于学习"""
        imgOrgShape = img.shape
        img, effectArea, realh, realw = resizeUniform(img, self.netInputSizehw)

        hsRate = realh / imgOrgShape[0]
        wsRate = realw / imgOrgShape[1]
        bboxes[:, 0:4:2] *= wsRate
        bboxes[:, 0] += effectArea['x']
        bboxes[:, 1:4:2] *= hsRate
        bboxes[:, 1] += effectArea['y']

        if self.showFlag: self.__show(np.copy(img).astype(np.uint8), bboxes, classes, self.annNames[index]+"_resize", color=(0, 0, 255))

        if self.augFlag :
            """Img Aug With Shape, 放射变换的增强一定要放在前面，主要是0的情况"""
            bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:] # (x1,y1, w,h)->(x1,y1, x2,y2)
            imgauger = ImgAugWithShape(img, bboxes)
            imgauger.shear(15)
            imgauger.translate(translate=[-0.2, 0.2])
            img, bboxes = (imgauger.img, imgauger.boxes)

            bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]  # (x1,y1, x2,y2)->(x1,y1, w,h)

            """非放射变换，放在最后， 最后的img 不用clip到（0，1）之间"""
            imgauger = ImgAugWithoutShape(img)
            imgauger.brightness()
            imgauger.constrast()
            imgauger.saturation()
            imgauger.normalize1(mean = self.normalize[0], std= self.normalize[1])
            img = imgauger.img

            if self.showFlag: self.__show(np.copy(img).astype(np.uint8), bboxes, classes, self.annNames[index] + "_aug", color=(0, 0, 255))
            if self.showFlag: cv2.waitKey()

        """return 两种return可供选择"""
        img = img.transpose(2, 0, 1)  # 因为pytorch的格式是CHW
        meta = dict(images=torch.from_numpy(img.astype(np.float32)),
                    bboxesGt=bboxes,
                    classes=classes,
                    annoName = self.annNames[index])
        #"""如果每个img输出的形状一样， 那么就可以下面"""
        # return torch.from_numpy(img.astype(np.float32)), \
        #        torch.from_numpy(bboxes.astype(np.float32)),\
        #        torch.from_numpy(classes.astype(np.float32))
        # 每张图片的bbox数目不一致，因此要使用这样使用, 在使用数目不一致的元素，就不能batch是使用，只能一个照片一张照片
        # torch.utils_less.data.DataLoader(trainData, collate_fn=collate_function,)这个需要collate_fn就需要制定一下了
        return meta

    def __len__(self):
        return len(self.annNames)

    def __show(self, img, bboxes,classes, winName, color):
        assert bboxes.shape[0] == classes.shape[0], "bboxes number not equal classes number!"
        for i in range(bboxes.shape[0]):
            cv2.rectangle(img, tuple((int(bboxes[i][0]), int(bboxes[i][1]))),
                          tuple((int(bboxes[i][0]) + int(bboxes[i][2]),
                                 int(bboxes[i][1]) + int(bboxes[i][3]))),
                          color, 1)
        for j in range(classes.shape[0]):
            cv2.putText(img, str(classes[j]), (int(bboxes[j][0]), int(bboxes[j][1])), 1, 1, color)
        cv2.imshow(winName, img)
if __name__ == '__main__':
    pass
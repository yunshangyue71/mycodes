import numpy as np
import os
import torch
from torch.utils.data import Dataset
import cv2
from data.imgaug_wo_shape import ImgAugWithoutShape
from data.imgaug_w_shape import ImgAugWithShape
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
                 trainAnnoPath,  # txt files root /
                 trainImgPath,  # images root /
                 netInputSizehw,
                 imgChannelNumber,
                 augFlag=False,
                 clsname = {0: "person"}
                 ):

        self.trainAnnoPath = trainAnnoPath
        self.trainImgPath = trainImgPath
        self.netInputSizehw = tuple(netInputSizehw)
        self.annNames = os.listdir(self.trainAnnoPath) # format me#["2008_000176.txt"]
        self.imgChannelNumber = imgChannelNumber
        self.augFlag = augFlag
        self.clsname = clsname
        self.showFlag = 0

    def __getitem__(self, index):
        """bbox img org"""
        txtPath = self.trainAnnoPath + self.annNames[index]

        """load infos"""
        infos = np.loadtxt(txtPath)
        if infos.ndim == 1:
            rows = infos.shape[0]
            infos = infos.reshape(-1, rows) #one row to 2dim

        """change int to float"""
        infos = np.array(infos, dtype=np.float32)

        """判断是不是背景图片"""
        if (infos ==np.array([[-1,-1,-1,-1,-1]])).all():
            bgFlag = True
        else:
            bgFlag = False

        bboxes = infos[:, :4]
        classes = infos[:, 4]

        """input img rgb or gray"""
        if self.imgChannelNumber == 3:
            img = cv2.imread(self.trainImgPath + self.annNames[index].split('.')[0] + '.jpg')# cv2.COLOR_BGR2RGB)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.imgChannelNumber == 1:
            img = cv2.imread(self.trainImgPath + self.annNames[index].split('.')[0] + '.jpg', cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32)

        winName = ""#self.annNames[index]
        # if self.showFlag:
        #     self.__show(np.copy(img).astype(np.uint8), bboxes, classes, winName, color = (0, 0, 255))

        """unifor resize 放在最后，输入网络的图片会有很多的0， 经过imgaug这些将会变为非0有利于学习"""
        img, infos, bboxes = resizeUniform(img, self.netInputSizehw, bboxes)
        if self.showFlag:
            self.__show(np.copy(img).astype(np.uint8), bboxes, classes, winName+"_resize", color=(0, 0, 255))

        """data shape augment"""
        if self.augFlag:
            """Img Aug With Shape, 放射变换的增强一定要放在前面，主要是0的情况"""
            bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:] # (x1,y1, w,h)->(x1,y1, x2,y2)
            if bgFlag:
                imgauger = ImgAugWithShape(img, None)
            else:
                imgauger = ImgAugWithShape(img, bboxes)
            imgauger.shear(5, prob =0.3)
            imgauger.translate(translate=0.1, prob=0.3)
            if not bgFlag:
                img, bboxes = (imgauger.img, imgauger.boxes)
            else:
                img = imgauger.img
            bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]  # (x1,y1, x2,y2)->(x1,y1, w,h)
            if self.showFlag:
                self.__show(np.copy(img).astype(np.uint8), bboxes, classes, winName + "_augshape", color=(0, 0, 255))

        """data color augment"""
        if self.augFlag:
            """非放射变换，放在最后， 最后的img 不用clip到（0，1）之间"""
            imgauger = ImgAugWithoutShape(img)
            imgauger.brightness(delta = 0.1, prob = 0.5)
            imgauger.constrast(alphaLow=0.9, alphaUp=1.1, prob = 0.5)
            imgauger.saturation(alphaLow=0.1, alphaUp=1.1, prob = 0.5)
            #imgauger.normalize1(mean = self.normalize[0], std= self.normalize[1])
            img = imgauger.img
            if self.showFlag:
                self.__show(np.copy(img).astype(np.uint8), bboxes, classes, winName + "_augcolor", color=(0, 0, 255))

        """see out put size"""
        if self.showFlag :
            outwh = (7,7)
            self.__show(np.copy(cv2.resize(img,(outwh[0], outwh[1]))).astype(np.uint8),
                        bboxes, classes, winName + "_augoutlayer",
                        color=(0, 0, 255))
        if self.showFlag:
            print(self.annNames[index])
            cv2.waitKey()
            #cv2.destroyAllWindows()

        # make dim == 3
        if self.imgChannelNumber == 1:
            img = img[:, :, np.newaxis]

        if bgFlag:
            bboxes = np.array([[-1,-1,-1,-1]])
        """return 两种return可供选择"""
        img = img.transpose(2, 0, 1)  # 因为pytorch的格式是CHW
        meta = dict(images=torch.from_numpy(img.astype(np.float32)),
                    bboxesGt=bboxes.astype(np.float32),
                    classes=classes.astype(np.float32),
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
            # print("")
        for i in range(bboxes.shape[0]):
            cv2.rectangle(img, tuple((int(bboxes[i][0]), int(bboxes[i][1]))),
                          tuple((int(bboxes[i][0]) + int(bboxes[i][2]),
                                 int(bboxes[i][1]) + int(bboxes[i][3]))),
                          color, 1)
            cv2.circle(img, tuple((int(bboxes[i][0]) + int(bboxes[i][2]/2),
                                 int(bboxes[i][1]) + int(bboxes[i][3]/2))),
                          2,color, -1)
        for j in range(classes.shape[0]):
            cv2.putText(img, self.clsname[int(classes[j])], (int(bboxes[j][0]), int(bboxes[j][1]+20)), 1, 1, color)
        cv2.imshow(winName, img)

if __name__ == '__main__':
    pass
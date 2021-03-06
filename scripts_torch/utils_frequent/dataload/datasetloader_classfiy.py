import numpy as np
import torch
from torch.utils.data import Dataset
import json
import cv2
"""
a dataset has a anno.json
[imgpath, cls]

example:cifar_10
"""
"""config """
trainAnnoPath = '/media/q/deep/me/data/cifar/cifar10_train.json'
trainImgPath = '/media/q/deep/me/data/cifar/cifar_train/'
testAnnoPath = '/media/q/deep/me/data/cifar/cifar10_test.json'
testImgPath = '/media/q/deep/me/data/cifar/cifar_test/'
netInputSize = (32, 32)
"""END """


class ListDataset(Dataset):
    def __init__(self, ):
        self.trainAnnoPath = trainAnnoPath
        self.trainImgPath = trainImgPath

        self.netInputSize = netInputSize

        with open(self.trainAnnoPath, 'r') as f:
            self.dict = json.load(f)

    def  __getitem__(self, index):
        img = cv2.imread(self.trainImgPath + self.dict[index][0])
        img = cv2.resize(img, self.netInputSize)
        label = np.array(self.dict[index][1])

        # 因为pytorch的格式是CHW
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img.astype(np.float32)), \
               torch.from_numpy(label.astype(np.float32))

    def __len__(self):
        return len(self.dict)

class ListDatasetTest(Dataset):
    def __init__(self, ):
        self.testAnnoPath = testAnnoPath
        self.testImgPath = testImgPath

        self.imgsize = (32, 32)

        with open(self.testAnnoPath, 'r') as f:
            self.dict = json.load(f)

    def  __getitem__(self, index):
        img = cv2.imread(self.testImgPath + self.dict[index][0])
        img = cv2.resize(img, self.imgsize)
        label = np.array(self.dict[index][1])

        # 因为pytorch的格式是CHW
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img.astype(np.float32)), \
               torch.from_numpy(label.astype(np.float32))

    def __len__(self):
        return len(self.dict)

if __name__ == '__main__':
    train_data = ListDataset()
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,  # 如果机器计算能力好的话，就可以设置为True，
    )

    print(len(train_loader))

    #下面是一个epoch，可以检验一下，一个epoch是不是等于batch迭代数目*batch size
    i = 0
    for imgs, targets in train_loader:

        i += 1
    print(i*32)

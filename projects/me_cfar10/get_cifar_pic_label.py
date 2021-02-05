import cv2
import numpy as np
import json

def _unpickle( fileDir):
    import pickle
    with open(fileDir, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        return dict

#pickleDir:me_cfar10 每个pickle的路径
#desDir：cifar10存放照片路径
def GetPic(pickleDir, desDir):
    #pickleDir = 'D:\project\\me_cfar10\\cifar-10-batches-py\\data_batch_%d'%(i+1)
    dict = _unpickle(pickleDir)
    imgNum = len(dict[b'labels'])
    print('imgs num: ', imgNum)

    for i in range(imgNum):
        img_ = dict[b'data'][i]
        img = img_.reshape((3,32,32)).transpose((1,2,0))
        cv2.imwrite( desDir + dict[b'filenames'][i].decode('utf-8'), img)


if __name__ == '__main__':
    #是否将cifar 的训练集进行unpickle
    trianFlag = 0
    if trianFlag:#unpicle train
        for i in range(5):
            pickleDir = '/media/q/deep/me/data/cifar/cifar-10-batches-py/data_batch_%d' % (i + 1)
            desDir = '/media/q/deep/me/data/cifar/cifar_train/'
            GetPic(pickleDir, desDir)

    #是否将cifar的测试集进行unpickle
    testFlag = 0
    if testFlag:
        pickleDir = '/media/q/deep/me/data/cifar/cifar-10-batches-py/test_batch'
        desDir = '/media/q/deep/me/data/cifar/cifar_test/'
        GetPic(pickleDir, desDir)

    #生成训练集的json annotation
    trainAnnoFlag = 0
    if trainAnnoFlag:
        cifarBatchRoot = '/media/q/deep/me/data/cifar/cifar-10-batches-py/'
        jsonDir = '/media/q/deep/me/data/cifar/cifar10_train.json'
        labels = []
        imgNames = []
        for i in range(5):
            pickleDir = cifarBatchRoot + 'data_batch_%d' % (i + 1)
            dict = _unpickle(pickleDir)
            imgNum = len(dict[b'labels'])
            print('imgs num: ', imgNum)

            labels.extend(dict[b'labels'])
            imgNames.extend(dict[b'filenames'])
        imgNames = [name.decode('utf-8') for name in imgNames]
        train = [[imgNames[i], labels[i]] for i in range(len(labels))]

        with open(jsonDir, "w") as f:
            json.dump(train, f)
            print("加载入文件完成...")

    # 生成测试集的json annotation
    testAnnoFlag = 0
    if testAnnoFlag:
        cifarBatchRoot = '/media/q/deep/me/data/cifar/cifar-10-batches-py/'
        jsonDir = '/media/q/deep/me/data/cifar/cifar10_test.json'
        labels = []
        imgNames = []

        pickleDir = cifarBatchRoot + 'test_batch'
        dict = _unpickle(pickleDir)
        imgNum = len(dict[b'labels'])
        print('imgs num: ', imgNum)

        labels.extend(dict[b'labels'])
        imgNames.extend(dict[b'filenames'])
        imgNames = [name.decode('utf-8') for name in imgNames]
        train = [[imgNames[i], labels[i]] for i in range(len(labels))]

        with open(jsonDir, "w") as f:
            json.dump(train, f)
            print("加载入文件完成...")

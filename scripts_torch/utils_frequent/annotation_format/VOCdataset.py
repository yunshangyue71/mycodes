import cv2
import numpy as np
from xml.etree import ElementTree as ET
"""get voc annosList
给定VOC中的一个任务， 然后从总的annos中抽出含有该任务的图片
用这个替代 dataList 中的 annoNames
"""
def vocAnnoPathes(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        annoNames = [line.strip().split(" ")[0] + ".xml" for line in lines]
    return annoNames

def vertify(xmin, ymin,xmax,ymax,width,height):
    assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
    assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
    assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
    assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

"""
在一个xml中解析出来的所有的box
annoPath: xml的路径
choiceCls：对那个类别进行检测
"""
def parseVoc(annoPath, choiceCls = ('person')):
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    LAYOUT = ("hand", "head", "")
    indexMap = dict(zip(CLASSES, range(len(CLASSES))))

    tree = ET.parse(annoPath)
    root = tree.getroot()
    size=root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    label = []
    for obj in root.iter("object"):
        difficult = int(obj.find("difficult").text)
        clsName = obj.find("name").text.strip().lower()
        if clsName not in choiceCls:
            continue
        clsId = indexMap[clsName]

        xmlbox = obj.find("bndbox")
        xmin = (float(xmlbox.find('xmin').text))
        ymin = (float(xmlbox.find('ymin').text) )
        xmax = (float(xmlbox.find('xmax').text) )
        ymax = (float(xmlbox.find('ymax').text) )
        try:
            vertify(xmin, ymin,xmax,ymax,width,height)
        except AssertionError as e:
            raise RuntimeError("Invalid label at {}, {}".format(annoPath, e))

        label.append([xmin, ymin, xmax-xmin, ymax-ymin, 0])
    if len(label) == 0:
        label = np.array([0, 0, 0, 0, -1])
    label = np.array(label)
    return label
if __name__ == '__main__':
    annotxtPath = "/media/q/data/datasets/VOC/VOC2012/ImageSets/Main/person_train.txt"
    imgRoot = "/media/q/data/datasets/VOC/VOC2012/JPEGImages/"
    annoRoot = "/media/q/data/datasets/VOC/VOC2012/Annotations/"

    annSaveDir = "/media/q/data/datasets/VOC/VOC2012/format_me/Main/person/train/"

    anns = vocAnnoPathes(annotxtPath)
    print(len(anns))
    annNames = []
    idx = 0
    for i in range(len(anns)):
        imgPath = imgRoot + anns[i].strip().split(".")[0] + ".jpg"
        annoPath = annoRoot + anns[i]
        #img = cv2.imread(imgPath)
        label = parseVoc(annoPath)

        # for j in range(label.shape[0]):
        #     img = cv2.rectangle(img, (int(label[j][0]), int(label[j][1])),
        #                         (int(label[j][0] + label[j][2]), int(label[j][1] + label[j][3])), (0,0,255), thickness=1, lineType=None, shift=None)
        # img = cv2.putText(img, CLASSES[int(label[j][4])], (int(label[j][0]), int(label[j][1]) + 10),
        #                   0, 1, (0,0,255), thickness=2, lineType=None, bottomLeftOrigin=None)
        if label.ndim > 1:
            annSavePath = annSaveDir + anns[i].split(".")[0] + ".txt"
            np.savetxt(annSavePath, label)

            print(idx)
            idx += 1


        # cv2.imshow("", img)
        # cv2.waitKey()



import cv2
import numpy as np
from xml.etree import ElementTree as ET
import os
import shutil

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
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

def parseVoc(annoPath, choiceCls = ['boat']):
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
        if difficult==1:
            continue
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
        if len(choiceCls) == 1:
            clsId =0
        label.append([xmin, ymin, xmax-xmin, ymax-ymin, clsId])
    if len(label) == 0:
        label = np.array([[-1, -1, -1, -1, -1]])
    label = np.array(label)
    return label
if __name__ == '__main__':
    """ config"""
    root = "/media/q/data/datasets/VOC/"
    dataName = "VOC2007_test"
    trainvaltest = "test.txt"#2007test:test; 2012:trainval

    # names = [CLASSES[14]] # only chocie one cls to  label, this label is 0
    # names = CLASSES # i want train 20 models, so every class has a dir,
    names = ["all"]# chocie all 20 class to label

    bg = 10# if chocie one class , bg% background will choice
    allNum = 16500 # 2012:16500; 2007test :5000
    diff1 = 1 #if  chocie difficult1,1:chocie diffi0 and diff 1; 0:only diffi0
    """end"""

    imgRoot = root + dataName  + "/JPEGImages/"
    annoRoot = root + dataName +"/Annotations/"

    for j in range( len(names)):
        name = names[j]

        if name == "all":
            dirname = "trainval_diff" + str(int(diff1))
            annotxtPath = root + dataName + "/" + "ImageSets/Main/"+trainvaltest
        else:
            dirname = "trainval_diff" + str(int(diff1)) + "_" + str(bg)+"bg"
            annotxtPath = root + dataName + "/" + "ImageSets/Main/"+name+"_" + trainvaltest

        annSaveDir = root + dataName + "/format_me/Main/" + name + "/" + dirname + "/"
        if os.path.exists(annSaveDir):
            shutil.rmtree(annSaveDir)
        if not os.path.exists(annSaveDir):
            os.makedirs(annSaveDir)

        anns = vocAnnoPathes(annotxtPath)
        print(j, "-"*10, len(anns))
        annNames = []
        idx = 0
        for i in range(len(anns)):
            imgPath = imgRoot + anns[i].strip().split(".")[0] + ".jpg"
            annoPath = annoRoot + anns[i]
            #img = cv2.imread(imgPath)
            if name == "all":
                label = parseVoc(annoPath, choiceCls=list(CLASSES))
            else:
                label = parseVoc(annoPath, choiceCls=[name])

            # for j in range(label.shape[0]):
            #     img = cv2.rectangle(img, (int(label[j][0]), int(label[j][1])),
            #                         (int(label[j][0] + label[j][2]), int(label[j][1] + label[j][3])), (0,0,255), thickness=1, lineType=None, shift=None)
            # img = cv2.putText(img, CLASSES[int(label[j][4])], (int(label[j][0]), int(label[j][1]) + 10),
            #                   0, 1, (0,0,255), thickness=2, lineType=None, bottomLeftOrigin=None)
            # if label.ndim > 1:

            if name!= "all" and bg > 0:
                dir0name = "trainval_diff" + str(int(diff1)) + "_" + str(0) + "bg"
                bg0dir = root + dataName + "/format_me/Main/" + name + "/" + dir0name + "/"
                assert os.path.exists(bg0dir), "0% background txt dir cannot be found ,so cannot add bg% background"
                objnum = len(os.listdir(bg0dir))

            if name=="all" or bg ==0:
                flag = False
            else:
                flag = np.random.random() < 0.1*objnum/(allNum-objnum)

            if not (label == np.array([[-1, -1, -1, -1, -1]])).all() or flag:# not bk  :#or(not flag1 and flag2):
                annSavePath = annSaveDir + anns[i].split(".")[0] + ".txt"
                np.savetxt(annSavePath, label)
            if idx%2000==0:
                print(idx)
            idx += 1


            # cv2.imshow("", img)
            # cv2.waitKey()



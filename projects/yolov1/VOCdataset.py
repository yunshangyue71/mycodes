import cv2
from xml.etree import ElementTree as ET

"""get voc annosList
用这个替代 dataList 中的 annoNames
"""
def vocAnnos(path):
    #path = "/media/q/data/datasets/VOC/VOC2012/ImageSets/Main/person_trainval.txt"
    with open(path, 'r') as f:
        lines = f.readlines()
        annoNames = [line.strip().split(" ")[0] + ".xml" for line in lines]
    return annoNames

"""
a xml
"""
def vocBox(annoPath):
    tree = ET.parse(annoPath)
    root = tree.getroot()
    for
    obj = root.iter("object")
    try:
        person = obj.find("person")

        bndbox = obj.find("bndbox")
        x1 = int(bndbox.find("xmin").text)
        y1 = int(bndbox.find("ymin").text)
        x2 = int(bndbox.find("xmax").text)
        y2 = int(bndbox.find("ymax").text)
        c = 0
    except:
        x1 = 0; y1 = 0
        x2 = 0; y2 = 0
        c = 0

    # return [x1, y1, x2-x1, y2-y1, c]
    # for child in root:
    #     print(child.tag, child.attrib, child.text)
    # print("done")
    raise

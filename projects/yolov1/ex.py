import os
import numpy as np
import cv2
import json

root = "/media/q/data/datasets/COCO/coco2017/"
tv = "val"
labeljson = root + "annotations_trainval2017/annotations/person_keypoints_"+tv + "2017.json"
imgDir = root +tv + "2017/"
with open(labeljson, 'r') as f:
    root = json.load(f)
    imgs = root['images']
    imgNum = len(imgs)
    annos = root['annotations']

for i in range(imgNum):
    imgid = imgs[i]["id"]
    imgName = imgs[i]["file_name"]
    imgpath = imgDir + imgName
    img = cv2.imread(imgpath)
    for j in range(len(annos)):
        imgidInanno = annos[j]["image_id"]
        boxes = []
        if imgidInanno == imgid:
            box = annos[j]["bbox"]
            # keypoint = annos[j]["keypoints"]
            # if int(annos[j]["iscrowd"]) != 0:
            #    color = (0,0,255)
            # else:
            #     color = (255, 0,0)
            # cv2.rectangle(img, (int(box[0]),int(box[1])),(int(box[0]+box[2]),int(box[1]+box[3])), color,2)
            # for k in range(17):
            #     cv2.circle(img,(int(keypoint[3*k]),int(keypoint[3*k+1])), 1 ,color, thickness=2 )
            boxes.append(box)
    np.savetxt("/media/q/data/datasets/COCO/coco2017/annotations_trainval2017/annotations/personbbox/" + tv + "/" +imgName.split(".")[0]+".txt",
                         boxes)
    if i % 100 ==0:
        print(i)

    # cv2.imshow(imgName, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
print("DONe")
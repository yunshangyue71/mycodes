import numpy as np
import cv2
import json

#
# path = '/media/q/deep/me/data/m2nist/m2nist/combined.npy'
saveDir = '/media/q/deep/me/data/m2nist/images/'
# imgs = np.load(path)
#
# for i in range(imgs.shape[0]):
#     cv2.imshow('a', imgs[i])
#     cv2.imwrite(saveDir + str(i).zfill(5)+'.jpg', imgs[i])
#     #cv2.waitKey()

annoDir = '/media/q/deep/me/data/m2nist/annotation/annotation_me/'
bboxPath = '/media/q/deep/me/data/m2nist/annotation/bbox.txt'
with open(bboxPath, 'r') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        infos = lines[i].strip().split('\t')
        name = infos[0].zfill(5)+'.txt'
        #img = cv2.imread(saveDir+name)
        bboxes = []
        for j in range(1, len(infos)):
            bbox = infos[j].split(',')
            bbox = [int(z) for z in bbox]
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]
            bboxes.append(np.array(bbox))
            # cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:4]), (0,0,255), 2)
            # cv2.imshow('img', img)
            # cv2.waitKey()
        bboxes = np.array(bboxes)
        np.savetxt(annoDir+name, bboxes)

a = np.loadtxt(annoDir+'00000.txt')
print(a)
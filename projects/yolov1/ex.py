import cv2
import os
import numpy as np

root = "/media/q/data/datasets/helmet/helm/"
mode = "valid/"
mask = 1

imgdir = root + "images/" + mode
annodir = root + "labels/" + mode

annNames = os.listdir(annodir)
annoSaveDir = root+"format_me/"+mode
if not os.path.exists(annoSaveDir):
    os.mkdir(annoSaveDir)
for i in range(len(annNames)):
    annoName = annNames[i]
    imgName = annoName.split(".")[0] + ".jpg"
    img = cv2.imread(imgdir + imgName)
    h, w, c = img.shape
    box = np.loadtxt(annodir + annoName)
    box = box.reshape((-1, 5))
    boxsave = np.copy(box)
    boxsave[:, 0:4] = box[:, 1:5]
    boxsave[:, 4] = box[:, 0]

    boxsave[:, :2] -= boxsave[:,2:4]/2
    # boxsave[:, 2:4] += boxsave[:, :2]
    boxsave[:, 0:3:2] *= w
    boxsave[:,1:4:2] *=h
    np.savetxt(annoSaveDir + annoName, boxsave)

    if i%1000==0:
        print(i)
    # boxwrite = []
    # for j in range(boxsave.shape[0]):
    #     cv2.rectangle(img, (int(boxsave[j][0]),int(boxsave[j][1]) ),
    #                   (int(boxsave[j][2]+boxsave[j][0]),int(boxsave[j][3]+boxsave[j][1]) ),
    #                   (0,255,0))
    #     if mask:
    #         ind = boxsave[:, ]
    # print(boxsave)
    # cv2.imshow("", img)
    # cv2.waitKey()
    # # h, w ,_ = img.shape

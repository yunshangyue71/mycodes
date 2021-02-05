#show
import numpy as np
from anchorbox_generate import AnchorGenerator
from anchorbox_assigner import Assigner
import torch
import cv2

"""需要配置， 一般默认就好"""
featSizes = [5, 4]#(h,w)
imageSize = (850, 680, 3)#(h,w)
boxesGt = np.array([[20,90,150,380],
                    [80,55,580,530],
                    [345, 345,515,515],
                    [170,510,340,675]])
classesGt = np.array([2,3, 4, 5])
"""END"""

"""画图片和anchor"""
img = np.zeros(imageSize)
lineRowStride = imageSize[1] / featSizes[1]
lineColStride = imageSize[0] / featSizes[0]
for i in range(featSizes[1] + 1):
    cv2.line(img, (int(lineRowStride * i), 0), (int(lineRowStride * i), imageSize[0]),
             color=(255, 255, 255), thickness=1)
    for j in range(featSizes[1] + 1):
        cv2.circle(img, (int(lineRowStride * i + lineRowStride/2),
                         int(lineColStride * j + lineColStride/2)),
                   color = (255,255,255), thickness=-1, radius=1)
for i in range(featSizes[0] + 1):
    cv2.line(img, (0, int(lineColStride * i)), (imageSize[1], int(lineColStride * i)),
             color=(255, 255, 255), thickness=1)

"""画anchor boxes"""
anchors = AnchorGenerator(
                baseSize = 30,  # size length basic
                scales = [5],  # [1., 10., 15.,] to make 3 kinds of boxes, w and h * scales
                hratios = [1],  # [1,2,3,10.5] , wratios = [1, 1/2, 1/3, 1/10, 1/5] to make 5 kinds of boxes
                scaleMajor=True,  # when times , first multiply scale
            ).gridAnchors(featmapSize = (featSizes[0], featSizes[1]), stride=170)
anchorsnp = anchors.to('cpu').numpy()
print('anchor boxes shape:', anchors.shape)
for i in range(anchorsnp.shape[0]):
    cv2.rectangle(img, (int(anchorsnp[i][0]),int(anchorsnp[i][1])),
                  (int(anchorsnp[i][2]), int(anchorsnp[i][3])),
                  color = (0,0,255), thickness=1)

"""画gtboxes"""
for i in range(boxesGt.shape[0]):
    cv2.rectangle(img, (int(boxesGt[i][0]),int(boxesGt[i][1])),
                  (int(boxesGt[i][2]), int(boxesGt[i][3])),
                  color = (0,255,0), thickness=1)
    txt = 'boxid:' + str(i) + '-' + 'cls:'+str(classesGt[i])
    cv2.putText(img,txt, (int(boxesGt[i][0]),int(boxesGt[i][1])),
                fontFace=1,color=(0,255,0),thickness=1, fontScale=1)

boxesGt = torch.from_numpy(boxesGt).to('cuda')
classesGt = torch.from_numpy(classesGt).to('cuda')
assign = Assigner(3, anchors, boxesGt, classesGt)
anchorBoxIndexGtBoxImg, anchorBoxGtClassImg = assign.master()

cv2.imshow('anchors boxes', img)

"""anchor box labeling"""
anchorsBoxGtClass = anchorBoxGtClassImg.to('cpu').numpy()
anchorBoxIndexGtBox = anchorBoxIndexGtBoxImg.to('cpu').numpy()
for i in range(anchorsBoxGtClass.shape[0]):
    if anchorsBoxGtClass[i] != -1:
        cv2.rectangle(img, (int(anchorsnp[i][0]), int(anchorsnp[i][1])),
                      (int(anchorsnp[i][2]), int(anchorsnp[i][3])),
                      color=(255, 0, 0), thickness=1)
        txt = 'boxid:'+str(anchorBoxIndexGtBox[i]) + '-' + 'cls:'+str(anchorsBoxGtClass[i])
        cv2.putText(img, txt,
                    (int(anchorsnp[i][0]),int(anchorsnp[i][1])),
                    fontFace=1,color=(255,0,0),thickness=1, fontScale=1)

cv2.imshow('assigned', img)
cv2.waitKey()
cv2.destroyAllWindows()
print('Done!')
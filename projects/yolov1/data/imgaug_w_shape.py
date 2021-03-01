import numpy as np
import cv2
import random
import math
"""
M = 
[[a   b   c]
 [d   e   f]
 [g   h   i]]
relate to center is (0, 0)
ori img = 
[[x],
 [y],
 [z]] 
a,e,i make the xyz isolate larger
b, make the x larger is the y is larger
c           x               z
d           y               x
f           y               z
g           z               x
h           z               y

the image is 2D , so the Z is a constant
"""
class ImgAugWithShape(object):
    def __init__(self, img, boxes):
        self.img = img
        self.boxes = boxes
        self.oriHeight = img.shape[0]  # shape(h,w,c)
        self.oriWidth = img.shape[1]

        #matrix make center (0, 0)
        self.C = np.eye(3)
        self.C[0, 2] = - self.oriWidth / 2
        self.C[1, 2] = - self.oriHeight / 2

        #matrix make p1 is (0, 0)
        self.T = np.eye(3)
        self.T[0, 2] = self.oriWidth / 2 # x translation
        self.T[1, 2] = self.oriHeight / 2 # y translation

    def fliplr(self, prob = 0.5):
        P = np.eye(3)
        if random.random() < prob:
            P[0, 0] = -1

        M = P @ self.C
        M = self.T @ M
        self.img = cv2.warpPerspective(self.img, M, dsize=None)
        self.boxes = self.warpBoxes(M)

    def flipud(self, prob=0.5):
        P = np.eye(3)
        if random.random() < prob:
            P[1, 1] = -1

        M = P @ self.C
        M = self.T @ M
        self.img = cv2.warpPerspective(self.img, M, dsize=None)
        self.boxes = self.warpBoxes(M)

    #degree du
    def rotation(self, degree, prob = 0.5):
        R = np.eye(3)
        if random.random() < prob:
            a = random.uniform(degree, degree)
            R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=1)

        M = R @ self.C
        M = self.T @ M
        self.img = cv2.warpPerspective(self.img, M, dsize=None)
        self.boxes = self.warpBoxes(M)

    #从中心放大 缩小图片，保持最终的尺寸不变, padding zero /cut
    def scale(self, ratio=(0.8, 1.2), prob = 0.5):
        Scl = np.eye(3)
        if random.random() < prob:
            scale = random.uniform(*ratio)
            Scl[0, 0] *= scale
            Scl[1, 1] *= scale

        M = Scl @ self.C
        M = self.T @ M
        self.img = cv2.warpPerspective(self.img, M, dsize=None)
        self.boxes = self.warpBoxes(M)

    # width, height scale different num
    def stretch(self, width_ratio=(0.8, 1), height_ratio=(0.8, 1), prob=0.5):
        Str = np.eye(3)
        if random.random() < prob:
            Str[0, 0] *= random.uniform(*width_ratio)
            Str[1, 1] *= random.uniform(*height_ratio)

        M = Str @ self.C
        M = self.T @ M
        self.img = cv2.warpPerspective(self.img, M, dsize=None)
        self.boxes = self.warpBoxes(M)

    #镜头畸变, padding zero
    def shear(self, degree=3, prob = 0.5):
        Sh = np.eye(3)
        if random.random() < prob:
            Sh[0, 1] = math.tan(random.uniform(degree, degree) * math.pi / 180)  # x shear (deg)
            Sh[1, 0] = math.tan(random.uniform(degree, degree) * math.pi / 180)  # y shear (deg)

        M = Sh @ self.C
        M = self.T @ M
        self.img = cv2.warpPerspective(self.img, M, dsize=None)
        self.boxes = self.warpBoxes(M)

    #up donw remove translate pixes, padding zero
    def translate(self, translate=0.2, prob=0.5):
        T = np.eye(3)
        if random.random() < prob:
            T[0, 2] = random.uniform(-translate, +translate) * self.oriWidth  # x translation
            T[1, 2] = random.uniform(-translate,  +translate) * self.oriHeight  # y translation

        M = T @ self.C
        M = self.T @ M
        self.img = cv2.warpPerspective(self.img, M, dsize=None)
        self.boxes = self.warpBoxes(M)

    def warpBoxes(self, M):
        n = len(self.boxes)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = self.boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, self.oriWidth)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, self.oriHeight)
            return xy.astype(np.float32)
        else:
            return self.boxes
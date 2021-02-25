import numpy as np
import cv2
import random
class ImgAugWithoutShape(object):
    def __init__(self, img):
        self.img = img

        pass
    def brightness(self, delta = 0.2, prob = 0.5):
        if random.uniform(0, 1) < prob:
            self.img += random.uniform(-delta, delta)
        return self.img

    def constrast(self, alphaLow=0.8, alphaUp=1.2, prob = 0.5):
        if random.uniform(0, 1) < prob:
            self.img *= random.uniform(alphaLow, alphaUp)
        return self.img

    def saturation(self, alphaLow=0.8, alphaUp=1.2, prob = 0.5):
        if random.uniform(0, 1) < prob:
            hsvImg = cv2.cvtColor(self.img.astype(np.float32)/255, cv2.COLOR_RGB2HSV)
            hsvImg[..., 1] *= random.uniform(alphaLow, alphaUp)
            self.img = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR) * 255
            # cv2.imshow('img', img/255)
        return self.img

    def normalize0(self, mean, std):
        self.img = self.img.astype(np.float32)
        mean = np.array(mean, dtype=np.float64).reshape(1, -1)
        stdinv = 1 / np.array(std, dtype=np.float64).reshape(1, -1)
        cv2.subtract(self.img, mean, self.img)
        cv2.multiply(self.img, stdinv, self.img)
        return self.img

    def normalize1(self, mean, std):
        mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255
        std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255
        self.img = (self.img - mean) / std
        return self.img


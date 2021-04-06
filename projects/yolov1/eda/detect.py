from config.config import load_config, cfg

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

class EDA(object):
    def __init__(self,imgdir, anndir):
        self.imgdir = imgdir
        self.anndir = anndir

        self.imgpaths =[os.path.join(imgdir, name) for name in os.listdir(imgdir)]
        self.annopaths = [os.path.join(anndir, name) for name in os.listdir(anndir)]
        self.imgname = imgdir.split('/')[-2] +imgdir.split('/')[-1]
        self.annoname = anndir.split("/")[-2]+anndir.split("/")[-1]
    def imgwh(self):
        ws = []
        hs = []
        cs = []
        fig = plt.figure(figsize = (16,16))
        for i in range(len(self.imgpaths)):
            img = cv2.imread(self.imgpaths[i])
            if img.ndim == 2:
                w, h = img.shape
                c = 1
            else:
                w,h,c = img.shape
            ws.append(w)
            hs.append(h)
            cs.append(c)
        ax1 = fig.add_subplot(221)
        ax1.hist(ws)
        ax1.set_title("img width")
        ax2 = fig.add_subplot(222)
        ax2.hist(hs)
        ax2.set_title("img hight")
        ax3 = fig.add_subplot(223)
        ax3.hist(cs)
        ax3.set_title("img channel")
        ax4 = fig.add_subplot(224)
        ax4.scatter(hs,ws)
        ax4.set_title("img width and hight")
        plt.savefig('./imgwh_'+self.imgname)
        plt.show()
    def boxwh(self):
        ws = []
        hs = []
        clss=[]

        areas = []
        larges = []
        mediums = []
        smalls = []
        imgboxnums = []
        fig = plt.figure(figsize=(16, 16))
        for i in range(len(self.annopaths)):
            box = np.loadtxt(self.annopaths[i])
            box = box.reshape((-1, 5))
            boxnum = box.shape[0]
            imgboxnums.append(boxnum)
            for j in range(boxnum):
                try:
                    ws.append(box[j][2])
                except:
                    print("")
                hs.append(box[j][3])
                clss.append(box[j][4])
                area = box[j][2]*box[j][3]
                areas.append(area)
                if area< 32*32:
                    smalls.append(area)
                elif area > 96*96:
                    larges.append(area)
                else:
                    mediums.append(area)

        ax1 = fig.add_subplot(3,3,1)
        ax1.hist(ws)
        ax1.set_title("box width")
        ax2 = fig.add_subplot(3,3,2)
        ax2.hist(hs)
        ax2.set_title("box hight")
        ax3 = fig.add_subplot(3,3,3)
        ax3.hist(clss)
        ax3.set_title("box class")
        ax4 = fig.add_subplot(3,3,4)
        ax4.scatter(hs, ws)
        ax4.set_title("img width and hight")

        ax5 = fig.add_subplot(3, 3, 5)
        ax5.hist(areas)
        ax5.set_title("area"+ str(len(areas)))

        ax6 = fig.add_subplot(3, 3, 6)
        ax6.hist(larges)
        ax6.set_title("large" + str(len(larges)))

        ax7 = fig.add_subplot(3, 3, 7)
        ax7.hist(mediums)
        ax7.set_title("medium" + str(len(mediums)))

        ax8 = fig.add_subplot(3, 3, 8)
        ax8.hist(smalls)
        ax8.set_title("smalls" + str(len(smalls)))

        ax9 = fig.add_subplot(3, 3, 9)
        ax9.hist(imgboxnums)
        ax9.set_title("boxes num per img" + str(len(imgboxnums)))


        plt.savefig('./box_' + self.annoname)
        plt.show()

if __name__ == '__main__':

    load_config(cfg, "../config/config.yaml")
    print(cfg)
    eda = EDA(cfg.dir.trainImgDir, cfg.dir.trainAnnoDir)
    # eda.imgwh()
    eda.boxwh()

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
            if i %1000 ==0:
                print(i,"-",len(self.imgpaths))
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
        ax4.scatter(hs,ws, s=1)
        ax4.set_title("img width and hight")
        plt.savefig('./imgwh_'+self.imgname)
        plt.show()
    def boxwh(self, clsname, boxnummax, resizeShapehw=None):
        """

        :param clsname: categray id : name
        :param boxnummax: when cal the box numper img, when boxnum beyound this , will teat as this
        :return:
        """
        clsbox = {} # box is this cls
        for k, v in clsname.items():
            clsbox[clsname[k]] = 0

        imgboxnums = {}
        for i in range( boxnummax):
            imgboxnums[i] = 0

        ws = []
        hs = []

        areas = []
        larges = []
        mediums = []
        smalls = []
        fig = plt.figure(figsize=(20,20))
        for i in range(len(self.annopaths)):
            if i % 1000 == 0:
                print(i, "-", len(self.annopaths))

            box = np.loadtxt(self.annopaths[i])
            box = box.reshape((-1, 5))
            boxnum = box.shape[0]
            if boxnum < boxnummax:
                imgboxnums[boxnum] +=1
            else:
                imgboxnums[boxnummax-1] +=1

            for j in range(boxnum):
                ws.append(box[j][2])
                hs.append(box[j][3])
                clsbox[clsname[int(box[j][4])]] +=1

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
        i=0
        for k, v in clsbox.items():
            ax3.barh(i*1.2, v, height = 1, label=v)
            ax3.text(v +1, i*1.2-0.5,  str(k) + ":" + str(v) + " %.2f"%(100*float(v)/len(areas))+"%")
            i += 1
        ax3.set_title("box class")

        ax4 = fig.add_subplot(3,3,4)
        ax4.scatter(hs, ws, s=1)
        ax4.set_title("box width and hight")

        ax5 = fig.add_subplot(3, 3, 5)
        ax5.hist(areas)
        ax5.set_title("area: total num is "+ str(len(areas)))

        ax6 = fig.add_subplot(3, 3, 6)
        ax6.hist(larges)
        ax6.set_title("large: area > 96*96, num is " + str(len(larges)))

        ax7 = fig.add_subplot(3, 3, 7)
        ax7.hist(mediums)
        ax7.set_title("medium: 32*32 < area < 96*96,num is " + str(len(mediums)))

        ax8 = fig.add_subplot(3, 3, 8)
        ax8.hist(smalls)
        ax8.set_title("smalls: area < 32*32, num is " + str(len(smalls)))

        ax9 = fig.add_subplot(3, 3, 9)
        i = 0
        for k, v in imgboxnums.items():
            ax9.barh(k, v, height=0.8, label=v)
            ax9.text(v + 1, k - 0.5, str(k) + ":" + str(v) + " %.2f"%(100*float(v)/len(self.annopaths))+"%")
            i += 1
        ax9.set_title("boxes num per img")


        plt.savefig('./box_' + self.annoname)
        plt.show()

if __name__ == '__main__':

    clsname=   {0: "head", 1: "helmet"}
    trainAnnoDir = "/media/q/data/datasets/helmet/helm/format_me/train/"
    trainImgDir = "/media/q/data/datasets/helmet/helm/images/train/"
    eda = EDA(trainImgDir, trainAnnoDir)
    eda.imgwh(resizeShapehw=(448, 448))
    # eda.boxwh(clsname = clsname, boxnummax=10)

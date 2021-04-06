import os
import shutil

dirr = "/media/q/data/datasets/VOC/VOC127/ImageSets/Segmentation/"
imgnames = os.listdir(dirr)
# print(len(imgnames))

imgdir7 = "/media/q/data/datasets/VOC/VOC2007/ImageSets/Segmentation/"
imgnames7 = os.listdir(imgdir7)
# print(len(imgnames7))

for i in range(len(imgnames7)):
    with open(imgdir7 + imgnames7[i], "r") as f7:
        lines7 = f7.readlines()
        print(len(lines7))
    with open(dirr + imgnames7[i], "r") as f1:
        lines = f1.readlines()
        print(len(lines))
    with open(dirr + imgnames7[i], "a+") as f:
        f.writelines(lines7)

    with open(dirr + imgnames7[i], "r") as f1:

        lines = f1.readlines()
        print(len(lines))
        print("---"*10)
    # oldfile = imgdir7 + imgnames7[i]
    # newfile = dirr + imgnames7[i]
    # shutil.copy(oldfile, newfile)
    # if i%100 == 0:
    #     print(i)
#
# iterr = 0
# for i in range(len(imgnames)):
#     name = imgnames[i]
#     if int(name.strip().split("_")[0]) == 2007:
#         iterr += 1
#     if name in imgnames7:
#         print(name)
# print(iterr)
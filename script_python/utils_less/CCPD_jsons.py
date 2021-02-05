"""
ccpd 车牌数据集 - 标注制作成json格式，方便自己统一
"""
import os
import json
root =  "/media/q/deep/me/data/CCPD2019_car_plate/CCPD2019/"
folder = "ccpd_weather"
imgdir = root + folder + '/'

imgNames = os.listdir(imgdir)
iter = 0
for imgName in imgNames[:]:
    # print(imgNames[0])
    # imgName = "0216654693486-89_89-267&474_524&563-521&561_265&562_266&475_522&474-0_0_18_16_27_30_24-141-106.jpg"
    infos = imgName.split('-')
    x1y1x2y2 = infos[2]
    x1y1,x2y2 = x1y1x2y2.split('_')
    x1, y1 = x1y1.split('&')
    x2, y2 = x2y2.split('&')
    bbox  = [int(i) for i in [x1, y1, x2, y2]]
    bbox[2] = bbox[2]-bbox[0]
    bbox[3] = bbox[3]-bbox[1]

    nums =infos[4].split('_')
    cls = [int(i) for i in nums]
    #print(bbox, cls)
    anno = []
    #anno.append(imgName)
    #anno.append(1)
    anno.extend(bbox)
    anno.extend(cls)
    #print(anno)
    with open(root+'annotations/'+folder+'_anno/' + imgName.split('.')[0]+'.json', 'w') as f:
        json.dump([anno],f)

    iter += 1
    if iter % 100 == 0:
        print(iter)

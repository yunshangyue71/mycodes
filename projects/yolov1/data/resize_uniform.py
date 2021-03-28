import cv2

#src :输入的图片 尺寸；dst：最终输出的图片的尺寸，已经pad了； real：实际图片缩放的后的尺寸
# bboxes: xywh
def resizeUniform(imgSrc, dstShape, bboxes=None):
    shapeSrc = imgSrc.shape
    hs, ws = shapeSrc[0], shapeSrc[1]
    hd, wd = dstShape
    ratios = ws * 1.0 / hs
    ratiod = wd * 1.0 / hd
    if ratios > ratiod:
        tmpw = wd
        tmph = int(wd / ratios)

        top = int((hd - tmph) / 2)
        down = (hd - tmph) -top
        left = right = 0

        effectArea = {'x': 0, "y": top, "w": tmpw, "h": tmph}

    else:
        tmph = hd
        tmpw = int(hd * ratios)

        top = down = 0
        left = int((wd - tmpw) / 2)
        right = (wd - tmpw) - left
        effectArea = {'x': left, "y": 0, "w": tmpw, "h": tmph}

    imgOut = cv2.resize(imgSrc, (int(tmpw),  int(tmph)))
    imgPad = cv2.copyMakeBorder(imgOut, int(top), int(down), int(left), int(right),
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])

    #bboxes
    if bboxes is not None:
        hsRate = tmph / hs
        wsRate = tmpw / ws
        bboxes[:, 0:4:2] *= wsRate
        bboxes[:, 0] += effectArea['x']
        bboxes[:, 1:4:2] *= hsRate
        bboxes[:, 1] += effectArea['y']

    #infos
    infos = {"effectArea": effectArea, "realh":tmph, "realw":tmpw}
    return imgPad, infos, bboxes# bbox resize will use

if __name__ == '__main__':
   imgpath = '/media/q/deep/me/data/WiderPerson/Images/000040.jpg'
   img = cv2.imread(imgpath)
   print(img.shape)
   imgout, ea, realh, realw = resizeUniform(img, (813,850))
   cv2.imshow("before ", img)
   print(realh, realw)
   cv2.imshow("", imgout)
   cv2.waitKey()
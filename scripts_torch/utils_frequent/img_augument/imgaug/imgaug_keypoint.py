
#https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B02%20-%20Augment%20Bounding%20Boxes.ipynb
import imageio
import imgaug as ia
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
%matplotlib inline

#Macropus_rufogriseus_rufogriseus_Bruny="https://upload.wikimedia.org/wikipedia/commons/e/e6/Macropus_rufogriseus_rufogriseus_Bruny.jpg"
Macropus_rufogriseus_rufogriseus_Bruny="D:/project/me_cfar10/imgaug_data/Macropus_rufogriseus_rufogriseus_Bruny.jpg"
image = imageio.imread(Macropus_rufogriseus_rufogriseus_Bruny)
image = ia.imresize_single_image(image, (389, 259))
#ia.imshow(image)

#keypoint位置
kps = [
    Keypoint(x=99, y=81),   # left eye (from camera perspective)
    Keypoint(x=125, y=80),  # right eye
    Keypoint(x=112, y=102), # nose
    Keypoint(x=102, y=210), # left paw
    Keypoint(x=127, y=207)  # right paw
]

#得到keypoint on image 的信息， 经过图像的增广，他们的相对位置不变， 必须图片对应
kpsoi = KeypointsOnImage(kps, shape=image.shape)
ia.imshow(kpsoi.draw_on_image(image, size=7))
print(kpsoi.keypoints)#可以打印图片的keypoint 位置信息

#keypoint 可以进行平移
kpsoi_pad = kpsoi.shift(x=100)
ia.imshow(kpsoi_pad.draw_on_image(image, size=7, color =(0, 255, 0)))


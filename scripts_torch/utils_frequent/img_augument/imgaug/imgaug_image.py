#%%

import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

%matplotlib inline

#%%

# 载入图片
#lena:"https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"

image = imageio.imread('D:\\project\\me_cfar10\\imgaug_data\\Lenna.png')
ia.imshow(image)

#%%

#对一张图片或者多张图片进行增广的操作

ia.seed(7)
rotate = iaa.Affine(rotate=(-25, 25))
image_aug = rotate(image=image)
ia.imshow(image_aug)

images = [image, image, image, image]
images_aug = rotate(images = images)
ia.imshow(np.hstack(images_aug))

#%%

#组合增广
#输入images的图片尺寸可以不同

seq = iaa.Sequential([
    #iaa.Affine(rotate=(-25, 25)),
    #iaa.AdditiveGaussianNoise(scale=(10, 60)),
    #iaa.Crop(percent=(0.1, 0.2))#上下各裁剪0.1， 左右各裁剪0.2, keep_size = False， 这两个值必须相等
    iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),  # crop and pad images
    iaa.AddToHueAndSaturation((-60, 60)),  # change their color
    iaa.ElasticTransformation(alpha=90, sigma=9),  # water-like effect
    iaa.Cutout()  # replace one squared area within the image by a constant intensity value
],
random_order=True#表示顺序随机
)
images_aug = seq(images = images)
ia.imshow(np.hstack(images_aug))

#%%

#不常用操作

#在多个cpu上进行增广https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/A03%20-%20Multicore%20Augmentation.ipynb

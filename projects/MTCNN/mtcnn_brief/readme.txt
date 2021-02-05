第一步：训练数据集的获得
  下载wider_face数据集，用于pos,part,neg图片的获得。只用了wider_face _trian这个里面的图片
  下载lfw_5590这个数据集（论文中使用的是celeba这个数据集，但是开源程序使用的是这个数据集）。下面的数据集中的train和test中的train，train中lfw和net中的lfw
  http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm
第二步：指定输出位置
  指定这个程序准备的图片以及输出的文件的根位置，在自己希望的位置创建文件夹并命名
第三步：path_and_config配置
  将wider_face的位置，lfw_5590,指定输出的位置放到这个文本，
第四步：将txt文件放进mtcnn_brief
  trainImageList.txt 		从lfw5590那个网站下载的资料里有
  wider_face_train.txt		从wider_face那个网站下载的资料里有
  wider_face_train_bbx_gt.txt	从wider_face那个网站下载的资料里有
第五步：训练测试步骤



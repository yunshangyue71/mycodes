## 项目结构：  
	config:          配置项目  
	net:             构建网络  
	trian:           训练网络  
	test_apic:       推理一张照片  
	saved_model:     模型保存  
	其他的禁止存放， 全部归类到util_frequents里面去  
## 日志：
### 2021.3.28
#### 解决loss 是none的问题
	torch sqrt 可以是0， 但是对其求导就不能等于0了

#### box nfidence 低的问题，错误的地区，会有score比较高， sigmod归一化， 这样的损失在（-1，1）之间超出了之后相差不打
	调整系数
	 noobj: 0.05  obj: 1   cls: 1  conf: 5  box: 5   l2: 0.0005
	错误方法：  
		1、n object的cell那么confidence就会始终是1
	正确方法：  
		1、confidence偏低， 认为损失函数受背景区域影响比较大， 所以降低noobj的系数为0.05  
		2、confidence 损失比较大相比于box高， 所以讲confidence和box 权重相同都设置为5
		3、本想着14*14更精细，一些，但是感受野小了， 所以confidence 定位不准确。  
#### 好多时候手臂会检测到， 图像边会检测到
	这个可能因为数据中， 只漏出一个手来，也标记为一个人了，这个可以检测，但是损失的系数可以小一些， 
	因为只有手臂标记的为人的情况，只有人在边缘的情况，所以可能学习到了边缘的这个信息
	自己代码中没有进行iou作为置信度， 所以iou低的框也被选择出来了
	
	

### 2021.4.2
#### 分析为什么自己的精度低的原因
	1、自己没有在imageNet数据集上预训练可能会有影响  
	2、yolov1的模型参数260M， 与之相同的模型是resnet50（101是paper中提出来的性能VGG16烧好， 50稍微次一些） 
	3、所以自己的目标就是resnet50 map能到60
#### 后来发现， batchsize 可以变大到32 resnet18， resnet50也可以16了， 应该是解决掉Nan 那个问题 后的作用
#### 解决之前遗留问题
	和paper 有出入， 如果有物体就是1，没有物体就是0， 如果两个相同的obj在同一个gride 那么选择iou大的那个
		这个以后会改
	一个box， 对应多个predbox 中iou最大的那个，
	如果一个cell 被一个obj占用， 后续的obj只有比他大才能进去
	使用条件概率作为label
### 2021.4.3
####
	添加background这种图片， 来减少false positive
	添加的时候错了， bbox经过 letter box 后不是-1，-1，-1，-1了，所以无法用这个判断背景了
	false positive 还是很多， 将noobj 损失系数加大到0.5，之前是0.05
	发现好多false positive 	，noobj 改为0.1
#### 16  not overfit maybe something wrong
### 2021.4.6
	lr stage , can deel this problem , when this model trained well , the l2 is not big 
### train eval mode 
	train will not freeze BN, Dropout
### 待做 
	变为20类的detect
	map查看效果，val的脚本
	tensorboard 记录，将背景图片添加到训练集，
	代码优化， 测试一张图片的脚本
	 
	z zzzzzz
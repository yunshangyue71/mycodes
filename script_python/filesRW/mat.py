import scipy.io as sio
import numpy as np
"""写入"""
## 将data变量保存在mat文件中data1对应的位置处
data = np.array([1, 2, 3, 5], dtype='float32')
sio.savemat('result.mat', {'data1': data})

"""读取"""
matPath = './egohands_data/metadata.mat'

'''探索一下数据格式'''
info = sio.whosmat(matPath)
print(info)

'''逐层探索'''
root = sio.loadmat(matPath)

#看第一层是不是字典， 如果是字典， 那么关键字有那些
keys = root.keys()
print(keys)
print(root["__header__"])
print(root["__version__"])
print(root["__globals__"])

#也可以看看是不是np的格式
video = root['video']
print("video shape: ", video.shape)

avideo = video[0][0]
print(len(avideo))
for i in range(len(avideo)-1):
    print(avideo[i])
avideoData = avideo[6][0]
for i in range(100):
    print(len(avideoData[i]))
#print(var)
# print(np.array(var)[0][0])
# print(var[1][0])


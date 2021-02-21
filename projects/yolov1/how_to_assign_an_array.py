#赋值
import torch
import numpy as np
a = torch.arange(24).reshape(2,3,4)
"""1、basic， 可以赋值， 可以广播赋值 """
a[1,2] = 100 # 下一维度自动广播
a[1, 2, 3] = 1000 # 没有广播
print("basic\n", a)
print("-"*20)

"""2、选择多个， 这里的0， 1表示该维度以及下一维度是否选择"""
b = torch.from_numpy(np.array([[0,0,0],
                               [0,0,0]]))
b = b.type(torch.bool)
print("select non\n",a[b])

b = torch.from_numpy(np.array([[1,1,1],
                               [0,0,0]]))
b = b.type(torch.bool)
print("select \n",a[b])
print("-"*20)

"""最后一个维度进行max，然后制作mask"""
maxNum, maxIndex = torch.max(a, dim = -1, keepdim=True)
class_num = 10
batch_size = 4
label = torch.LongTensor(batch_size, 7, 7, 1).random_() % class_num
one_hot = torch.zeros(batch_size,  7, 7,class_num).scatter_(-1, label, 1)
print("done")
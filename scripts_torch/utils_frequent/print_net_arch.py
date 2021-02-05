import netron
import torch

#查看结构图
net = FPN(cfg)
x = torch.rand(58,320,320)# no batchsize
net(x)
onnx_path = '/media/q/deep/me/model/pytorch_script_use/????'
torch.onnx.export(net, x, onnx_path)
netron.start(onnx_path)

#查看模型大小
from torchsummary import  summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = FPN().to(device)
summary(net, (2,58,320,320))
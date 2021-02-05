import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytorchModelPath = '/media/q/deep/me/project/script_torch/projects/nano_det/saved_model/100.pth'
onnxModelPath = '/media/q/deep/me/project/script_torch/projects/nano_det/saved_model/100.onnx'

from net import NanoNet
network = NanoNet(classNum=10)
network.to(device)

weights = torch.load(pytorchModelPath )#加载参数
network.load_state_dict(weights)#给自己的模型加载参数


# #set the model to inference mode
network.eval()

x = torch.randn(1, 3, 480, 320)   # 生成张量
x = x.to(device)

export_onnx_file = onnxModelPath		# 目的ONNX文件名
torch.onnx.export(network,
                    x,
                    export_onnx_file,
                    opset_version=11,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],	# 输入名
                    output_names=["output0b","output1b","output2b","output0c","output1c","output2c"],	# 输出名
                    dynamic_axes={"input":{0:"batch_size"},  # 批处理变量， 这个一般不设置，
                                    "output":{0:"batch_size"}}
	)

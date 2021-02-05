import onnx
from onnxsim import simplify

onnxModelPath = '/media/q/deep/me/project/script_torch/projects/nano_det/saved_model/100.onnx'
model = onnx.load(onnxModelPath)

model_simp, check = simplify(model)
assert check, "simplified Onnx model could not be validated"
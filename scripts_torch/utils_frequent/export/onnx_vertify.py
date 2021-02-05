#onnx——vertify
import numpy as np
import onnx
import onnxruntime as rt
import cv2

onnxModelPath = '/media/q/deep/me/project/script_torch/projects/nano_det/saved_model/100-sim.onnx'
imgpath ='/media/q/deep/me/data/m2nist/format_orgin/images/00000.jpg'

# create input data
# input_data = np.ones((1, 3, 32, 32), dtype=np.float32)

input_data = cv2.imread(imgpath)
input_data = cv2.resize(input_data, (480,320))
input_data = input_data.transpose(2, 0, 1)/255.0
input_data = np.array(input_data, dtype=np.float32).reshape((1,3,480,320))

# create runtime session
sess = rt.InferenceSession(onnxModelPath)

# get output name
input_name = sess.get_inputs()[0].name
print("input name", input_name)
output_names = sess.get_outputs()#[0]#.name
print("output nums", len(output_names))

for i in range(len(output_names)):
    output_name = sess.get_outputs()[i].name
    output_shape = sess.get_outputs()[i].shape
    print("output shape", output_shape)
    # forward model
    res = sess.run([output_name], {input_name: input_data})
    out = np.array(res).reshape((-1, 10))
    #print(out)

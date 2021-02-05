#test_apic_example
from AnchorFreeStructure import *
from torch.utils.data import DataLoader
from numBoxes import ListDataset
import cv2 as cv
from torch.autograd import Variable
import os
import warnings
import json
warnings.filterwarnings("ignore")

imgsize = (256, 256)
inputimg = np.zeros([1, 3, imgsize[0], imgsize[1]])
if __name__ == '__main__':
    network = body()
    network.load_state_dict(torch.load('./model/NumDetect_230000.pt', map_location=torch.device('cpu')))
    for parameter in network.parameters():
        parameter.requires_grad = False
    network.eval()
    
    imgDirs = []
    for imDir in imgDirs:
        im = cv.imread(imDir)
        h, w, _ = im.shape
        img = cv.resize(im, imgsize)
        inputimg[0, ...] = img.transpose(2, 0, 1).astype(np.float) / 255
        imgs = torch.from_numpy(inputimg).float()
        with torch.no_grad():
            #inference 结果的后处理
            detections = network(imgs).detach().numpy()
            conf = detections[0, 0, ...]
            classes = detections[0, 5:, :, :]
            boxes = detections[0, 1:5, ...]
            classes = classes
            boxesout = get_boxes([conf], [boxes], [classes])
        #结果的显示
        cv.imshow('', im)
        cv.waitKey(0)
        cv.destroyWindow('')

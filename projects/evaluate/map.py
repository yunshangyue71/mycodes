import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import json
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print ('Running demo for *%s* results.'%(annType))

#initialize COCO ground truth api
dataDir='/media/q/data/datasets/VOC/VOC2007_test/format_me/Main/person/'
dataType='train2017'
annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
annFile = "/media/q/data/datasets/VOC/VOC2007_test/format_me/Main/person/coco/annotations/instances_train2017.json"
cocoGt=COCO(annFile)

#initialize COCO detections api
resFile='%s/results/%s_%s_fake%s100_results.json'
resFile = resFile%(dataDir, prefix, dataType, annType)
resFile = "/media/q/data/datasets/VOC/VOC2007_test/format_me/Main/person/coco/annotations/instances_train20172.json"
with open(resFile) as f:
    anns = json.load(f)
resFile = anns['annotations']
cocoDt=cocoGt.loadRes(resFile)

imgIds=sorted(cocoGt.getImgIds())
imgIds=imgIds[0:100]
imgId = imgIds[np.random.randint(100)]

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
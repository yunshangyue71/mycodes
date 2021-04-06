# A PyTorch implementation of a YOLO v1 Object Detector
 Implementation of YOLO v1 object detector in PyTorch. Full tutorial can be found [here](https://deepbaksuvision.github.io/Modu_ObjectDetection/) in korean.

 Tested under Python 3.6, PyTorch 0.4.1 on Ubuntu 16.04, Windows10.

## Requirements

See [requirements](./requirements.txt) for details.

NOTICE: different versions of PyTorch package have different memory usages.

## How to use
### Training on PASCAL VOC (20 classes)
```
main.py --mode train -data_path where/your/dataset/is --class_path ./names/VOC.names --num_class 20 --use_augmentation True --use_visdom True
```

### Test on PASCAL VOC (20 classes)
```
main.py  --mode test --data_path where/your/dataset/is --class_path ./names/VOC.names --num_class 20 --checkpoint_path your_checkpoint.pth.tar
```

#### pre-built weights file
```python
python3 utilities/download_checkpoint.py
```
[pre-build weights donwload](https://drive.google.com/open?id=1lgpHENZm2HGhEVHAIX4mK_ATSU_ujeqy)

## Supported Datasets
Only Pascal VOC datasets are supported for now.

## Configuration Options
|argument          |type|description|default|
|:-----------------|:----|:---------------------- |:----|
|--mode            |str  |train or test           |train|
|--dataset         |str  |only support voc now    |voc  |
|--data_path       |str  |data path               |     |
|--class_path      |str  |filenames text file path|     |
|--input_height    |int  |input height            |448  |
|--input_width     |int  |input width             |448  |
|--batch_size      |int  |batch size              |16   |
|--num_epochs      |int  |# of epochs             |16000|
|--learning_rate   |float|initial learning rate   |1e-3 |
|--dropout         |float|dropout probability     |0.5  |
|--num_gpus        |int  |# of GPUs for training  |1    |
|--checkpoint_path |str  |checkpoint path         |./   |
|--use_augmentation|bool |image Augmentation      |True |
|--use_visdom      |bool |visdom                  |False|
|--use_wandb       |bool |wandb                   |False|
|--use_summary     |bool |descripte Model summary |True |
|--use_gtcheck     |bool |gt check flag           |False|
|--use_githash     |bool |use githash             |False|
|--num_class       |int  |number of classes       |5    |

## Train Log
![train_log](https://user-images.githubusercontent.com/13328380/50018219-dd5fd500-0011-11e9-9040-86222c91d5c6.png)

## Results 
![image](https://user-images.githubusercontent.com/15168540/49991740-61d83680-ffc5-11e8-8912-096033351060.png)
![image](https://user-images.githubusercontent.com/15168540/49991762-71f01600-ffc5-11e8-9b65-6e3aec0c7504.png)
![image](https://user-images.githubusercontent.com/15168540/49991795-86341300-ffc5-11e8-9d29-1ed601789bc4.png)
![image](https://user-images.githubusercontent.com/15168540/49991804-8cc28a80-ffc5-11e8-997d-f3a6a4a027fb.png)


## Authorship
This project is equally contributed by [Chanhee Jeong](https://github.com/chjeong530), [Donghyeon Hwang](https://github.com/ssaru), and [Jaewon Lee](https://github.com/insurgent92).

## Copyright
See [LICENSE](./LICENSE) for details.

## REFERENCES
[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. (https://arxiv.org/abs/1506.02640)

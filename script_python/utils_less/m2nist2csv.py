import numpy as np
import csv

bboxPath = '/media/q/deep/me/data/m2nist/annotation/bbox.txt'
bboxCSVPath = '/media/q/deep/me/data/m2nist/annotation/bbox.csv'
with open(bboxCSVPath,'w', newline='') as f:
    csvf = csv.writer(f)
    with open(bboxPath, 'r') as f1:
        lines = f1.readlines()
        for line in lines:
            infos = line.strip().split('\t')
            name = infos[0].zfill(5) + '.jpg'
            bboxes = []
            for j in range(1, len(infos)):
                bbox = infos[j].split(',')
                bbox = [int(z) for z in bbox]
                csvf.writerow([name, bbox[0], bbox[1], bbox[2],bbox[3],bbox[4]])

import numpy as np
import os

dirs =  "/media/q/deep/me/data/WiderPerson/Annotations/"
txtnames = os.listdir(dirs)
print(len(txtnames))
for i in range(len(txtnames)):
    path = dirs + txtnames[i]
    a = np.loadtxt(path,skiprows=1)
    if a.shape[0]==1:
        print(path)
        print(a.shape)
print("done")
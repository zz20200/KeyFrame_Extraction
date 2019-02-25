# coding= utf-8

import os
import numpy as np
import cv2

path1 = r"F:\pythonProgram\KeyFrame_11_29\dataBase\database2"
path2 = r"F:\pythonProgram\KeyFrame_11_29\dataBase\database3"



for i in range(1,5000):
    os.chdir(path1)
    name = str(i).zfill(6)+".jpg"
    im = cv2.imread(name)
    im2 = im[:, 260:980, :]
    os.chdir(path2)
    cv2.imwrite(name,im2)
    
    

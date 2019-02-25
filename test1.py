import cv2
import numpy as np
import random
import os
import copy

import CAE1_zz_1127
import key_frame_kmeans_time_VC_1_18


os.chdir(r"F:\pythonProgram\KeyFrame_11_29\tmp1")
# gray = np.zeros((252,252),np.uint8)
gray = cv2.imread("0-20.jpg",cv2.IMREAD_GRAYSCALE)
gray = gray[0:720, 280:1000]
gray = cv2.resize(gray, (252, 252), interpolation=cv2.INTER_AREA)

# 提取灰度图252*252的特征图56*56
feature = CAE1_zz_1127.try_extractfeature(gray)
feature = np.uint8(feature)
# matrix = np.reshape(feature, (-1, 1))
cv2.imwrite("test1.bmp",feature)



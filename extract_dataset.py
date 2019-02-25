import imageio
import os
import cv2
import random
import numpy as np
# 提取出00000-00004文件夹中，每个文件夹里随机取40个视频，每个视频中每隔5帧取一帧保存
n=1
rootdir = "F:\\SLR-DATA\\xf500_color_video"
for filename in os.listdir(rootdir)[:5]:
    print(filename)
    # 取前5个视频
    subdir = os.path.join(rootdir, filename)
    if(os.path.isdir(subdir)):
        for videoname in random.sample(os.listdir(subdir),40):
            videopath = os.path.join(subdir,videoname)

            cap = cv2.VideoCapture(videopath)
            # 获取FPS(每秒传输帧数(Frames Per Second))
            fps = cap.get(cv2.CAP_PROP_FPS)
            # 获取总帧数
            totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print(totalFrameNumber)
            # 当前读取到第几帧
            COUNT = 10
            # # 随机取两帧的图像进行保存
            # keyframe1 = random.randint(2, totalFrameNumber)
            # keyframe2 = random.randint(2, totalFrameNumber)
            # print(str(keyframe1)+" "+str(keyframe2))
            # 每隔5帧进行读取
            for location in range(8,int(totalFrameNumber-8),5):
                # 从loaction帧开始读取视频
                cap.set(cv2.CAP_PROP_POS_FRAMES, location)
                ret,frame = cap.read()

                # 截取一个正方形照片720*720*3
                frame_crop =frame[0:720,280:1000,:]
                # 转换为灰度图像并变成252*252
                gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (252, 252), interpolation=cv2.INTER_AREA)
                # 把每一帧图像保存成bmp格式（这一行可以根据需要选择保留）
                cv2.imwrite('C:\\shen\\random_extracted_img\\'+str(n)+ '.bmp', gray)
                n+=1






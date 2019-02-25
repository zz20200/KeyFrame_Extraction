import cv2
import numpy as np
import random
import os
import copy
import CAE1_zz_1127
import video2img

'''
传统K-means，取每类中最靠近质心的
'''

PATH_BASE = os.getcwd()
PATH_KeyFrame = os.path.join(PATH_BASE, r'key_frame')
# PATH_DB = os.path.join(PATH_BASE, 'dataBase')##todo:zhushi
PATH_DB = os.path.join(PATH_BASE,'tmp1')

def eu_distance(data_set, oneCentriod):
    result1 = data_set - oneCentriod
    result2 = np.multiply(result1, result1)
    result3 = np.mean(result2, axis=0)
    result4 = np.sqrt(result3)
    dis = result4.tolist()
    return dis


def KnnClass(dataSet, k):
    m, n = dataSet.shape
    ##step1:随机寻找k个样本作为初始质心
    indxK = random.sample(range(0, n), k)
    indxK = np.sort(indxK)
    centriodVec = dataSet[:, indxK]

    centriodSetLast = copy.deepcopy(centriodVec)
    ##step2:计算所有样本与各个质心的距离，比较大小，将其分类，并重新计算质心，直到满足条件
    while (1):
        ou_dis = []
        dis = [[] for dims in range(k)]
        for i in range(0, k):  # 对于k各质心，计算所有样本与之距离
            disKOu = eu_distance(dataSet, centriodVec[:, i:i + 1])  # 返回距离列表
            ou_dis.append(disKOu)

        dis = ou_dis

        indx = [[] for dim in range(k)]  # 创建一个K维列表，第i维代表第i个质心对应的样本的索引
        for j in range(0, n):  # 因为有n个样本， 对于每个样本与每个中心的距离，将其标记为选择最小的那一类
            kdisOu = []  ##初始化列表，长度为k，第i个元素表示某个样本与第i个质心的距离
            for kN in range(0, k):  # 对于第j个样本，它与每个质心的距离
                kdisOu.append(dis[kN][j])

            kdisOu = np.array(kdisOu)
            classNum = (np.argsort(kdisOu)).tolist()  # 返回的值是距离最小的下标索引，这个索引刚好表示这个样本属于第几类
            # classNum.reverse()
            indx[classNum[0]].append(j)

        ##计算新的质心向量
        for l in range(0, k):
            dataKclass = dataSet[:, indx[l]]
            centriodVec[:, l:l + 1] = (np.mean(dataKclass, axis=1)).reshape(m, 1)

        centriodVec = np.uint8(centriodVec)

        ##看新计算的质心是否改变（截至条件）
        flag = np.sum(np.fabs(centriodVec - centriodSetLast))
        centriodSetLast = copy.deepcopy(centriodVec)  # 更新
        print("flag", flag)
        if flag < 0.5:
            print("完成")
            for im in range(0, k):
                imThis = centriodVec[:, im]
                imThis2 = np.reshape(imThis, (56, 56))
                # cv2.imwrite(str(im) + ".jpg",imThis2)
            break
    return indx, centriodSetLast


def makeData(nameVideo):
    global num
    # os.chdir(PATH_VIDEO)
    cap = cv2.VideoCapture(nameVideo)

    # 获取总帧数
    totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    K = int( (totalFrameNumber-25) // 7)
    videomat = np.empty((3136, 1))
    COUNT = 1
    # 若小于总帧数则读一帧图像
    while COUNT < totalFrameNumber - 10: #去掉最后15帧

        # 一帧一帧图像读取
        ret, frame = cap.read()
        if ret == 0:
            break
        if (COUNT > 15): #去掉前面15帧
            # 剪裁图片并变灰度
            frame = frame[0:720, 280:1000, :]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (252, 252), interpolation=cv2.INTER_AREA)
            # 提取灰度图252*252的特征图56*56
            feature = CAE1_zz_1127.try_extractfeature(gray)
            feature = np.uint8(feature)
            matrix = np.reshape(feature, (-1, 1))
            videomat = np.c_[videomat, matrix]

        COUNT = COUNT + 1

    cap.release();
    index, centriodSetLast = KnnClass(videomat[:, 1:], K)
    print(index)
    videoMat2 = videomat[:, 1:]

    # os.chdir(PATH_VIDEO)
    cap = cv2.VideoCapture(nameVideo)
    for k in range(0, K):
        # 每个类里取离聚类中心最近的feature对应的那一帧
        dataThis = videoMat2[:, index[k]]
        centroidThis = centriodSetLast[:, k].reshape(-1, 1)

        disThis = eu_distance(dataThis, centroidThis)

        indexDisThis = np.argsort(disThis)

        indexNearestThis = index[k][indexDisThis[0]] + 1

        index1 = index[k][0]
        imThis = videomat[:, indexNearestThis]
        imThis2 = np.reshape(imThis, (56, 56))
        # 取视频中的第index1+10帧，因为取videomat时除去了视频的前10帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, (index1 + 10))
        ret, frame = cap.read()
        os.chdir(PATH_DB)
        # cv2.imwrite( str(k) + ".jpg", imThis2)
        cv2.imwrite(str(num).zfill(6) + ".jpg", frame)
        num += 1


global num
num = 1

rootdir = r"E:\dataBase\SLR-DATA\xf500_color_video"

for filename in os.listdir(rootdir)[151:152]:#todo：150-200
    print(filename)
    subdir = os.path.join(rootdir, filename)
    if (os.path.isdir(subdir)):
        # 每个文件夹随机取5个视频
        videoname = "P18_08_11_0_color.avi" # for videoname in random.sample(os.listdir(subdir), 5):#todo：注释掉
        videopath = os.path.join(subdir, videoname)
        print(videoname)
        video2img.vdeo2img(videopath,15,10)
        try:
            makeData(videopath)
        except:
            continue

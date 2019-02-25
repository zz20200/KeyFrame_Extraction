import cv2
import numpy as np
import random
import os
import copy
import time
import CAE1_zz_1127

'''
1.K-means with CAE
2.sort by time，选每个连续类中第一个
'''

def clearFolder(path):
    """
    删除文件夹下的所有文件，包括子文件夹中的文件
    """
    for i in os.listdir(path):
        path_file = os.path.join(path, i)  # 取文件绝对路径
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            clearFolder(path_file)

def sortByTime(index):
    # 按时间分开聚得每一类中的各个连续的部分帧
    global kf_before
    kf_before = []  # 作为不加清晰度选择的关键帧
    keyFrameIndex = []

    for idx in index:  # 遍历聚得的每一类idx
        # 如果该类只有一帧，直接加入keyFrameIndex
        if len(idx) == 1:
            keyFrameIndex.extend(idx)
            kf_before.extend(idx)
            continue

        dif = [idx[i] - idx[i - 1] for i in range(1, len(idx))]
        dif.insert(0, 2)  # 2是随便取的，只要不是1就行
        # 各连续部分的第一帧的索引位置
        dif = np.array(dif)
        firstframepos = np.where(dif != 1)
        firstframepos = firstframepos[0]
        # 如果idx中只有一个连续部分
        if len(firstframepos) == 1:
            continuousFrames = idx
            # 对当前continuousFrames做清晰度选择处理，选择其中最清晰的一帧
            # keyFrameIndex.append(CTCF(continuousFrames))
            kf_before.append(continuousFrames[0])

        else:
            for j in range(1, len(firstframepos)):
                # 取每两个第一帧索引之间的帧为连续的一部分
                continuousFrames = idx[firstframepos[j - 1]: firstframepos[j]]
                # 对当前continuousFrames做清晰度选择处理，选择其中最清晰的一帧
                # keyFrameIndex.append(CTCF(continuousFrames))
                kf_before.append(continuousFrames[0])
                if j == (len(firstframepos) - 1):
                    # 取每两个第一帧索引之间的帧为连续的一部分
                    continuousFrames = idx[firstframepos[j]: len(idx)]
                    # 对当前continuousFrames做清晰度选择处理，选择其中最清晰的一帧
                    # keyFrameIndex.append(CTCF(continuousFrames))
                    kf_before.append(continuousFrames[0])

    # 按时间从小到大排序
    kf_before.sort()
    # print("不做清晰度选择的关键帧"+str(kf_before))
    keyFrameIndex.sort()
    return keyFrameIndex

def eu_distance(data_set, oneCentriod):
    result1 = data_set - oneCentriod
    result2 = np.multiply(result1 , result1)
    result3 = np.mean(result2,axis = 0)
    result4 = np.sqrt(result3)
    dis = result4.tolist()
    return dis

def KnnClass(dataSet,k):
    m, n = dataSet.shape
    ##step1:随机寻找k个样本作为初始质心
    indxK = random.sample(range(0,n),k)
    indxK = np.sort(indxK )
    centriodVec = dataSet[:,indxK]


    centriodSetLast = copy.deepcopy(centriodVec)
    ##step2:计算所有样本与各个质心的距离，比较大小，将其分类，并重新计算质心，直到满足条件
    while(1):
        ou_dis = []
        dis = [[] for dims in range(k)]
        for i in range(0, k):           #对于k各质心，计算所有样本与之距离
            disKOu = eu_distance(dataSet,centriodVec[:,i:i+1])   #返回距离列表
            ou_dis.append(disKOu)

        dis = ou_dis

        indx = [[] for dim in range(k)]  #创建一个K维列表，第i维代表第i个质心对应的样本的索引
        for j in range(0,n):             #因为有n个样本， 对于每个样本与每个中心的距离，将其标记为选择最小的那一类
            kdisOu = []                  ##初始化列表，长度为k，第i个元素表示某个样本与第i个质心的距离
            for kN in range(0,k):#对于第j个样本，它与每个质心的距离
                kdisOu.append(dis[kN][j])

            kdisOu = np.array(kdisOu)
            classNum = (np.argsort(kdisOu) ).tolist()     #返回的值是距离最小的下标索引，这个索引刚好表示这个样本属于第几类
            # classNum.reverse()
            indx[ classNum[0]].append(j)

        ##计算新的质心向量
        for l in range(0,k):
            dataKclass = dataSet[:,indx[l]]
            centriodVec[:,l:l+1] = ( np.mean(dataKclass, axis = 1)).reshape(m,1)


        centriodVec = np.uint8(centriodVec)

        ##看新计算的质心是否改变（截至条件）
        flag =  np.sum ( np.fabs(centriodVec - centriodSetLast) )
        centriodSetLast = copy.deepcopy(centriodVec) #更新
        # print("flag",flag)
        if flag < 0.5:
            # print("完成")
            for im in range(0, k):
                imThis = centriodVec[:, im]
                # imThis2 = np.reshape(imThis, (56, 56))
                # cv2.imwrite(str(im) + ".jpg",imThis2)
            break
    return indx,centriodSetLast


def method1(nameVideo):

    cap = cv2.VideoCapture(nameVideo)
    # 获取总帧数
    totalFrameNumber = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("totalFrameNumber:" + str(totalFrameNumber))
    # 去掉frontcut和endcut的特征图矩阵videomat
    videomat = np.empty((3136, 1))
    COUNT = 1
    frontcut = 20
    endcut = 20
    global framemat
    framemat = []

    # 去掉最后endcut帧,读每一帧图像
    while COUNT < totalFrameNumber - endcut:

        # 一帧一帧图像读取
        ret, frame = cap.read()
        if ret == 0:
            break
        if COUNT > frontcut:  # 去掉前面frontcut帧
            # 剪裁图片并变灰度
            frame = frame[0:720, 280:1000, :]

            # os.chdir(path)
            # cv2.imwrite(str(COUNT - frontcut - 1).zfill(6) + '.jpg', frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            framemat.append(gray)
            gray = cv2.resize(gray, (252, 252), interpolation=cv2.INTER_AREA)
            # 提取灰度图252*252的特征图56*56
            feature = CAE1_zz_1127.try_extractfeature(gray)
            feature = np.uint8(feature)
            matrix = np.reshape(feature, (-1, 1))
            videomat = np.c_[videomat, matrix]

        COUNT = COUNT + 1

    cap.release();

    videomat = videomat[:, 1:]
    videomat2 = videomat[1568:, :]

    firstframe = videomat2[:, 0]
    firstframe = np.reshape(firstframe, (1568, 1))
    endframe = videomat2[:, totalFrameNumber - frontcut - endcut - 2]
    endframe = np.reshape(endframe, (1568, 1))

    # 检测与第一帧相似的帧有哪些
    similarFrame = []
    similarFrameIndex = []
    disKOu = eu_distance(videomat2, firstframe)
    # print("与第一帧的差距：")
    # print(disKOu)
    # print([abs(disKOu[i] - disKOu[i - 1]) for i in range(1, len(disKOu))])

    similarFrameIndex.append(0)
    for i in range(1, len(disKOu)):
        if abs(disKOu[i] - disKOu[i - 1]) > 1:
            similarFrame.append(disKOu[i - 1])
            similarFrameIndex.append(i)
        else:
            break

    # print(similarFrame)
    # print(similarFrameIndex)

    # 检测与最后一帧相似的帧有哪些
    similarFrame1 = []
    similarFrameIndex1 = []
    disKOu = eu_distance(videomat2, endframe)
    # print("与最后一帧的差距：")
    # print(disKOu)
    # print([abs(disKOu[i] - disKOu[i - 1]) for i in range(1, len(disKOu))])

    for i in reversed(range(1, len(disKOu))):
        if abs(disKOu[i] - disKOu[i - 1]) > 1:
            similarFrame1.append(disKOu[i - 1])
            similarFrameIndex1.append(i)
        else:
            break

    # print(similarFrame1)
    # print(similarFrameIndex1)

    # 剩下的帧的特征做一个矩阵newVideomat
    cutFrameIndex = similarFrameIndex + similarFrameIndex1
    global kmeansFrameIndex
    kmeansFrameIndex = set(range(0, totalFrameNumber - frontcut - endcut - 1)) - set(cutFrameIndex)
    kmeansFrameIndex = list(kmeansFrameIndex)
    # print(kmeansFrameIndex)
    newVideomat = []
    COUNT = 0

    while 1:

        if COUNT == len(videomat):
            break

        if COUNT in kmeansFrameIndex:
            # 提取灰度图252*252的特征图56*56
            m=framemat[COUNT]
            m=np.reshape(m,(1,-1))
            newVideomat.append(m[0])
        COUNT = COUNT + 1
    newVideomat = np.array(newVideomat)
    newVideomat = np.transpose(newVideomat)

    # 对newVideomat做聚类
    K = int(len(kmeansFrameIndex) // 6)  # todo:聚类的数量
    index, centriodSetLast = KnnClass(newVideomat, K)
    for i in range(len(index)):
        for j in range(len(index[i])):
            index[i][j] = index[i][j] + len(similarFrameIndex)
    # print(index)
    KF = sortByTime(index)
    print("方法1提取关键帧数：" + str(len(kf_before)))
    return totalFrameNumber,len(kf_before)

num_mean = 0
total_mean = 0
root = r"I:\dataset\000245"
for video in os.listdir(root):
    print(video)
    nameVideo = os.path.join(root, video)
    total,num = method1(nameVideo)
    total_mean = total_mean +total
    num_mean = num_mean + num
num_mean = num_mean/50
total_mean = total_mean/50 - 40
# 平均总帧数
print(total_mean)
# 方法1平均提取关键帧数
print(num_mean)

# method1(r"I:\dataset\000447\P25_23_07_0_color.avi")
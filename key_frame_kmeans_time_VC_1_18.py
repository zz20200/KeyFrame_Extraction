import cv2
import numpy as np
import random
import os
import copy
import CAE1_zz_1127
import time
from skimage import filters


'''my mythod'''

def get_median(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0:  # 判断列表长度为偶数
        median = (data[size // 2] + data[size // 2 - 1]) / 2
        data[0] = median
    if size % 2 == 1:  # 判断列表长度为奇数
        median = data[(size - 1) // 2]
        data[0] = median
    return data[0]


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


def SMD2Detection(img2gray):
    """
    灰度方差乘积
    """
    # 图片对象转化矩阵
    img2gray = np.matrix(img2gray)
    f = img2gray / 255.0
    x, y = f.shape
    score = 0
    for i in range(x - 1):
        for j in range(y - 1):
            score += np.abs(f[i + 1, j] - f[i, j]) * np.abs(f[i, j] - f[i, j + 1])

    return score


def BrenerDetection(img2gray):
    """
    Brenner梯度函数最简单的梯度评价函数指标，他只是简单的计算相邻两个像素灰度差的平方
    """
    # 图片对象转化矩阵
    img2gray = np.matrix(img2gray)
    f = img2gray / 255.0
    x, y = f.shape
    score = 0
    for i in range(x - 2):
        for j in range(y - 2):
            score += (f[i + 2, j] - f[i, j]) ** 2

    return score


def TenengradDetection(img2gray):
    """
    Tenengrad 梯度函数采用Sobel算子分别提取水平和垂直方向的梯度值，基与Tenengrad 梯度函数的图像清晰度定义如下
    """
    # 图片对象转化矩阵
    f = np.matrix(img2gray)

    tmp = filters.sobel(f)
    source = np.sum(tmp ** 2)
    score = np.sqrt(source)

    return score


def CTCF(cf_index):
    """
    choose the clearest frame from continuousFrames
    """
    global framemat
    scores = []
    for i in cf_index:
        # 可选择三种不同的清晰度方法:Tenengrad，Brener，SMD2,Vollath
        # t1 = time.clock()
        score = TenengradDetection(framemat[i])
        # t2 = time.clock()
        # print("SMD2清晰度检测每张耗时" + str(t2-t1))
        # print(score)
        scores.append(score)
    # 按降序排列scores，得到最大分数的索引
    scores = np.array(scores)
    i = np.argsort(-scores)[0]
    clearestFrame = cf_index[i]
    clearestFrame_score = scores[i]
    # print("最大清晰度的一帧为" +str(clearestFrame)+",分数为"+ str(clearestFrame_score))
    return clearestFrame


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
            keyFrameIndex.append(CTCF(continuousFrames))
            kf_before.append(continuousFrames[0])

        else:
            for j in range(1, len(firstframepos)):
                # 取每两个第一帧索引之间的帧为连续的一部分
                continuousFrames = idx[firstframepos[j - 1]: firstframepos[j]]
                # 对当前continuousFrames做清晰度选择处理，选择其中最清晰的一帧
                keyFrameIndex.append(CTCF(continuousFrames))
                kf_before.append(continuousFrames[0])
                if j == (len(firstframepos) - 1):
                    # 取每两个第一帧索引之间的帧为连续的一部分
                    continuousFrames = idx[firstframepos[j]: len(idx)]
                    # 对当前continuousFrames做清晰度选择处理，选择其中最清晰的一帧
                    keyFrameIndex.append(CTCF(continuousFrames))
                    kf_before.append(continuousFrames[0])

    # 按时间从小到大排序
    kf_before.sort()
    # print("不做清晰度选择的关键帧"+str(kf_before))
    keyFrameIndex.sort()
    return keyFrameIndex


def eu_distance(data_set, oneCentriod):
    result1 = data_set - oneCentriod
    result2 = np.multiply(result1, result1)
    result3 = np.mean(result2, axis=0)
    result4 = np.sqrt(result3)
    dis = result4.tolist()
    return dis


def KnnClass(dataSet, k):
    m, n = dataSet.shape
    # step1:随机寻找k个样本作为初始质心
    indxK = random.sample(range(0, n), k)
    indxK = np.sort(indxK)
    centriodVec = dataSet[:, indxK]

    centriodSetLast = copy.deepcopy(centriodVec)
    ##step2:计算所有样本与各个质心的距离，比较大小，将其分类，并重新计算质心，直到满足条件
    while (1):
        ou_dis = []
        # dis = [[] for dims in range(k)]
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
        # print("flag", flag)
        if flag < 0.5:
            # print("完成")
            for im in range(0, k):
                imThis = centriodVec[:, im]
                imThis2 = np.reshape(imThis, (56, 56))
                # cv2.imwrite(str(im) + ".jpg",imThis2)
            break
    return indx, centriodSetLast


def saveFrame(videopath, index, savepath, frontcut, endcut):
    # 读取视频
    cap = cv2.VideoCapture(videopath)
    # 获取总帧数
    totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # 当前读取到第几帧
    count = 0
    # 若小于总帧数则读一帧图像
    while count < totalFrameNumber - endcut:
        # 一帧一帧图像读取
        ret, frame = cap.read()

        if (count - frontcut) in index:
            # 把每一帧图像保存成jpg格式（这一行可以根据需要选择保留）
            os.chdir(savepath)
            cv2.imwrite(str(count - frontcut) + '.jpg', frame)

        count = count + 1

    cap.release();


def pointDensityImprove(index, interval=5):  # todo:修改interval

    """
    通过点密度二次优化关键帧
    """
    global kmeansFrameIndex
    # 几帧划分一个区间
    dis = [index[i] - index[i - 1] for i in range(1, len(index))]
    # pos =np.where(np.array(dis)<4)
    # print(pos[0])
    # keyf = []
    # for p in pos[0]:
    #     keyf.append(index[p + 1])
    # keyf = list(set(index) - set(keyf))
    # keyf.sort()
    dis_mean = np.mean(dis)
    density = []  # 对于每个index元素的点密度矩阵
    for i in range(0, len(index)):
        c = 0
        for x in index:
            # 以每个元素为原点，dis_mean为半径的区间内，计算index元素的个数c
            if (x < (index[i] + dis_mean)) & (x > (index[i] - dis_mean)):
                c += 1
        density.append(c)
    density_mean = np.mean(density)
    # density_median = get_median(density)
    # print("density:" + str(density) + "阈值:" + str(density_mean))
    # 区间划分的起点idx为kmeansFrameIndex[0]
    idx = kmeansFrameIndex[0]
    index = np.array(index)
    keyf = []
    while idx < index[len(index) - 1]:
        # 获取在每5帧的一个区间内的index元素下标
        pos = np.where((index >= idx) & (index < (idx + interval)))

        if not len(pos[0]):
            idx += interval
            continue

        pos_ = list(map(int, pos[0]))
        # den为该区间每个元素的点密度，ind为该区间的关键帧
        den = [density[i] for i in pos_]
        ind = [index[i] for i in pos_]
        # 获取在该区间点密度大于阈值density_mean的den中元素下标
        pos1 = np.where(np.array(den) > density_mean)  # todo：点密度阈值

        if not len(pos1[0]):
            idx += interval
            continue

        pos1_ = list(map(int, pos1[0]))
        # den为该区间在阈值以上的每个元素的点密度，ind为该区间的关键帧
        den1 = [den[i] for i in pos1_]
        ind1 = [ind[i] for i in pos1_]
        # 获取den1中第一个最大值的下标,并添加到keyf
        keyf.append(ind1[den1.index(max(den1))])

        idx += interval

    return keyf


def mymethod(nameVideo):
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

            os.chdir(path)
            cv2.imwrite(str(COUNT - frontcut - 1).zfill(6) + '.jpg', frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            framemat.append(gray) # 灰度图像矩阵
            gray = cv2.resize(gray, (252, 252), interpolation=cv2.INTER_AREA)
            # 提取灰度图252*252的特征图56*56
            feature = CAE1_zz_1127.try_extractfeature(gray)
            feature = np.uint8(feature)
            matrix = np.reshape(feature, (-1, 1))
            videomat = np.c_[videomat, matrix]# 特征矩阵

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
            newVideomat.append(videomat[:, COUNT])

        COUNT = COUNT + 1
    newVideomat = np.array(newVideomat)
    newVideomat = np.transpose(newVideomat)

    # 对newVideomat做聚类
    K = int(len(kmeansFrameIndex) // 6)  # todo:聚类的数量
    index, centriodSetLast = KnnClass(newVideomat, K)
    # print(index)
    # print('z')
    for i in range(len(index)):
        for j in range(len(index[i])):
            index[i][j] = index[i][j] + len(similarFrameIndex)
    # print(index)
    KF = sortByTime(index)
    # print("不做清晰度选择的关键帧" + str(kf_before))
    # print("经过清晰度筛选后按时间排序的关键帧：" + str(KF))
    # 去掉keyframe中相隔小于3的帧

    dis = [KF[i] - KF[i - 1] for i in range(1, len(KF))]
    pos = np.where(np.array(dis) < 3)# todo：帧间隔需大于2
    # print(pos[0])
    keyframe = []
    for p in pos[0]:
        keyframe.append(KF[p + 1])
    keyframe = list(set(KF) - set(keyframe))
    keyframe.sort()
    # 另一种方法
    # i=0
    # while i<len(KF)-1:
    #     if KF[i+1]-KF[i] <3:
    #         KF.remove(KF[i+1])
    #     else:
    #         i+=1
    # keyframe = KF
    # print("去掉KF中相隔小于3的帧" + str(keyframe))

    # savepath = r"F:\pythonProgram\KeyFrame_11_29\tmp1"
    # clearFolder(savepath)
    # # 保存第一次关键帧
    # saveFrame(nameVideo, keyframe, savepath, frontcut, endcut)

    # 二次优化keyframe
    keyframe_optimized = pointDensityImprove(keyframe)
    # print("二次优化筛选出的关键帧：" + str(keyframe_optimized))

    # 保存优化后的关键帧
    ko_savepath = r"I:\KeyFrame_11_29\key_frame"
    clearFolder(ko_savepath)
    saveFrame(nameVideo, keyframe_optimized, ko_savepath, frontcut, endcut)
    # end = time.clock()
    # print(end - start)
    print("本方法提取关键帧数：" + str(len(keyframe_optimized)))
    return totalFrameNumber,len(keyframe_optimized)

'''
去掉前后20帧
'''
'''
START
'''
# 清空video2img文件夹
path = r"I:\KeyFrame_11_29\video2img"
clearFolder(path)
# start = time.clock()
root = r"C:\Users\Administrator\Desktop\dataset\000245"# todo:视频路径
num_mean = 0
for video in os.listdir(root):
    print(video)
    nameVideo = os.path.join(root, video)
    total,num = mymethod(nameVideo)
    # total_mean = total_mean +total
    num_mean = num_mean + num
num_mean = num_mean/50
print(num_mean)
# nameVideo = r"E:\dataBase\SLR-DATA\public_dataset\color\000000\P01_s1_00_0_color.avi"
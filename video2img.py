import cv2
import os
'''
frontcut:视频最前面剪几帧
endcut:视频最后面剪几帧
'''
def vdeo2img(videopath,frontcut,endcut):
    # 读取视频
    cap = cv2.VideoCapture(videopath)
    # 获取FPS(每秒传输帧数(Frames Per Second))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取总帧数
    totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(fps)
    print(totalFrameNumber)
    # 当前读取到第几帧
    COUNT = 0

    # 若小于总帧数则读一帧图像
    while COUNT < totalFrameNumber-endcut:
        # 一帧一帧图像读取
        ret, frame = cap.read()
        if (COUNT > frontcut):
            # 把每一帧图像保存成jpg格式（这一行可以根据需要选择保留）
            os.chdir(r"F:\pythonProgram\KeyFrame_11_29\video2img")
            cv2.imwrite(str(COUNT - frontcut -1) + '.jpg', frame)
        # 显示这一帧地图像
        # cv2.imshow('video', frame)
        COUNT = COUNT + 1
        # 延时一段33ms（1s➗30帧）再读取下一帧，如果没有这一句便无法正常显示视频
        # cv2.waitKey(33)

    cap.release();

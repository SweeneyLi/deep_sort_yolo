# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 09:32:02 2016

@author: http://blog.csdn.net/lql0716
"""
import cv2
import numpy as np
from tqdm import tqdm, trange
import time

current_pos = None
tl = None
br = None


# 鼠标事件
def get_rect(im, title='get_rect'):  # (a,b) = get_rect(im, title='get_rect')
    mouse_params = {'tl': None, 'br': None, 'current_pos': None,
                    'released_once': False}

    cv2.namedWindow(title)
    cv2.moveWindow(title, 100, 100)

    def onMouse(event, x, y, flags, param):

        param['current_pos'] = (x, y)

        if param['tl'] is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
            param['released_once'] = True

        if flags & cv2.EVENT_FLAG_LBUTTON:
            if param['tl'] is None:
                param['tl'] = param['current_pos']
            elif param['released_once']:
                param['br'] = param['current_pos']

    cv2.setMouseCallback(title, onMouse, mouse_params)
    cv2.imshow(title, im)

    while mouse_params['br'] is None:
        im_draw = np.copy(im)

        if mouse_params['tl'] is not None:
            cv2.rectangle(im_draw, mouse_params['tl'],
                          mouse_params['current_pos'], (255, 0, 0))

        cv2.imshow(title, im_draw)
        _ = cv2.waitKey(10)

    cv2.destroyWindow(title)

    tl = (min(mouse_params['tl'][0], mouse_params['br'][0]),
          min(mouse_params['tl'][1], mouse_params['br'][1]))
    br = (max(mouse_params['tl'][0], mouse_params['br'][0]),
          max(mouse_params['tl'][1], mouse_params['br'][1]))

    return (tl, br)  # tl=(y1,x1), br=(y2,x2)


def readVideo(pathName):
    # print("Please input s to stop the video and choose the area with mouse!\nThen input q to the next step!")
    print("按s暂停视频，用鼠标选择区域，可重复选择\n按q结束")
    time.sleep(2)

    cap = cv2.VideoCapture(pathName)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    left_top, bottom_right = (0, 0), (width, height)
    frame_index = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            (left_top, bottom_right) = get_rect(frame, title='get_rect')

        cv2.namedWindow("Video", 0)
        cv2.resizeWindow('Video', 960, 540)
        cv2.imshow('frame', frame[left_top[1]:bottom_right[1], left_top[0]: bottom_right[0], :])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return (left_top, bottom_right), fps


def video_process(video_path, new_fps, speed_rate, new_position):
    (left_top, bottom_right) = new_position
    width = bottom_right[0] - left_top[0]
    height = bottom_right[1] - left_top[1]

    # show_video = input("Do you need to show the video ? Please input y or n! (default is y)\n")
    show_video = input("需要预览视频吗？按y或n，默认是")
    show_video = False if show_video == 'n' else True

    # output_path = input("Please input the output path! (default means same as input)\n")
    output_path = input("请输入输出路径(包含文件名）！默认保存原视频目录下!\n")
    if not output_path:
        video_name = video_path.split("\\")[-1]
        output_path = video_path.replace(video_name, video_name.split(".")[0] + "_processed." + "mov")

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), new_fps, (width, height))
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = 39 * fps
    # end_frame = (6 * 60 - 5)* fps + 1

    for i in trange(frame_cnt):
        ret, frame = cap.read()
        if i < start_frame:
            continue
        # if i > end_frame:
        #     break
        if i % speed_rate == 0:
            frame = frame[left_top[1]:bottom_right[1], left_top[0]: bottom_right[0], :]
            out.write(frame)
            if show_video:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 点击视频窗口，按q键退出
                    break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("The video is processed!\n The path is %s." % output_path)


# video_path = r"D:\video\B6_2020_5_27_1.mp4"
# video_path = r"D:\WorkSpaces\deep_sort_yolov3\output\past\r_DJI_0010.mov"
# 中文
video_path = input("请输入视频路径!\n")
import os
while not os.path.exists(video_path):
    video_path = input("该路径错误，请重新输入!\n")

(left_top, bottom_right), fps = readVideo(video_path)
print("原视频的fps为 %s" % fps)
try:
    speed_rate = input("请输入加速比，默认为3（直接回车）\n")
    speed_rate = 3 if speed_rate == '' else int(speed_rate)
    new_fps = input("请输入新的fps，默认为原fps/加速比（直接回车）\n")
    new_fps = fps / speed_rate if new_fps == '' else int(new_fps)
except Exception as e:
    print("输入格式错误!")
    print(e)

if speed_rate != 1 or fps != new_fps:
    video_process(video_path, new_fps, speed_rate, (left_top, bottom_right))
    print("转换完成!")
else:
    print("无需转换!")

# English
# video_path = input("Please input your video path!\n")
# import os
# while not os.path.exists(video_path):
#     video_path = input("The path is wrong! Please input again!\n")
#
# (left_top, bottom_right), fps = readVideo(video_path)
# print("The original fps of videos is %s" % fps)
# try:
#     speed_rate = input("Which speed rate do you want? Please input speed rate. (default is 3 )\n")
#     speed_rate = 3 if speed_rate == '' else int(speed_rate)
#     new_fps = input("Which fps do you want? Please input new fps. (default means same fps)\n")
#     new_fps = fps if new_fps == '' else int(new_fps)
# except Exception as e:
#     print("Please input the right format!")
#     print(e)
#
# if speed_rate != 1 or fps != new_fps:
#     video_process(video_path, new_fps, speed_rate, (left_top, bottom_right))
#     print("End!")
# else:
#     print("No need to change!")
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 09:32:02 2016

@author: http://blog.csdn.net/lql0716
"""
import cv2
import numpy as np
from tqdm import tqdm, trange
import time
import argparse
import os

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
    # time.sleep(2)

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
        # cv2.resizeWindow('Video', 960, 540)
        cv2.imshow('frame', frame[left_top[1]:bottom_right[1], left_top[0]: bottom_right[0], :])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return (left_top, bottom_right), fps


def video_process(video_path, new_fps, speed_rate, new_position, fill_position=None):
    # fill_position = np.array([
    #     [[1550, 0], [1882, 0], [1880, 400]]
    # ])
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

    # start_frame = (3 * 60 + 50) * fps
    # end_frame = (4 * 60 + 50) * fps
    start_frame = 0
    end_frame = None

    for i in trange(frame_cnt):
        ret, frame = cap.read()
        if i < start_frame:
            continue
        if not end_frame is None and i > end_frame:
            break
        if i % speed_rate == 0:
            frame = frame[left_top[1]:bottom_right[1], left_top[0]: bottom_right[0], :]

            if not fill_position is None:
                frame = cv2.fillPoly(frame, fill_position, (0, 0, 0))

            out.write(frame)
            if show_video:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 点击视频窗口，按q键退出
                    break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("The video is processed!\n The path is %s." % output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default=r"D:\Videos\5m-z.mov")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_path = args.video_path

    while not os.path.exists(video_path):
        video_path = input("该路径错误，请重新输入!\n")

    (left_top, bottom_right), fps = readVideo(video_path)
    print("原视频的fps为 %s" % fps)
    try:
        speed_rate = input("请输入加速比，默认为3（直接回车）\n")
        speed_rate = 3 if speed_rate == '' else int(speed_rate)
        new_fps = input("请输入新的fps，默认为原fps/加速比（直接回车）\n")
        new_fps = fps / speed_rate if new_fps == '' else int(new_fps)

        video_process(video_path, new_fps, speed_rate, (left_top, bottom_right))
    except Exception as e:
        print("输入格式错误!")
        print(e)

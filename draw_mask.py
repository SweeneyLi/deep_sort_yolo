import argparse
import json
import os

import cv2
import numpy as np

json_path = r"mask_points.json"
EXIT_COLOR = (66, 183, 42)
tpPointsChoose = []

MASK_LIST = []


def dram_the_picture():
    img = Original_Img.copy()
    for a_mask in MASK_LIST:
        a_mask = list(map(lambda p: [p[0] / factor_of_change, p[1] / factor_of_change], a_mask))
        mask = np.zeros(img.shape, np.uint8)
        pts = np.array([a_mask], np.int32)
        pts = pts.reshape((-1, 1, 2))

        cv2.polylines(mask, [pts], True, (100, 100, 100))
        cv2.fillPoly(mask, [pts], EXIT_COLOR)
        cv2.addWeighted(mask, 1, img, 1, 0, img)
    return img


def on_mouse(event, x, y, flags, param):
    global img
    global tpPointsChoose, MASK_LIST
    img = dram_the_picture()

    if event == cv2.EVENT_LBUTTONDOWN:
        x, y = change_point_border(x, y)
        cv2.circle(img, (x, y), 10, (0, 255, 0), 2)
        tpPointsChoose.append([x, y])

        for i in range(len(tpPointsChoose) - 1):
            cv2.line(img, tuple(tpPointsChoose[i]), tuple(tpPointsChoose[i + 1]), (0, 0, 255), 2)
        cv2.imshow('src', img)

    if event == cv2.EVENT_RBUTTONDOWN and len(tpPointsChoose) > 2:
        tpPointsChoose = list(
            map(lambda p: [int(p[0] * factor_of_change), int(p[1] * factor_of_change)], tpPointsChoose))
        MASK_LIST.append(tpPointsChoose)
        tpPointsChoose = []
        img = dram_the_picture()
        cv2.imshow('src', img)

    if event == cv2.EVENT_MBUTTONDOWN:
        MASK_LIST = []
        tpPointsChoose = []
        img = Original_Img
        cv2.imshow('src', img)


def change_point_border(x, y):
    # print((x, y), end="->")
    if x < diff_size:
        x = 0
    elif width - x < diff_size:
        x = width

    if y < diff_size:
        y = 0
    elif height - y < diff_size:
        y = height
    # print((x, y))
    return (x, y)


# def mask_list_change_factor(the_MASK_List, factor_of_change):
#
#     for a_mask in the_MASK_List:
#         for p in a_mask:
#             p[0] *= factor_of_change
#             p[0] = min(width, int(p[0]))
#             p[1] *= factor_of_change
#             p[0] = min(height, int(p[1]))
#     return the_MASK_List


# for showing the picture in the small screen
factor_of_change = 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default=r"D:\Videos\5m-z.mov")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_path = args.video_path

    print("操作说明：\n左键选择点，右键确定（将选择的点作为一个封闭的图形），\n保存‘名字：点’到mask_points.json中，\n中键清除该视频所有点的信息，标注完成后esc退出，")

    cap = cv2.VideoCapture(video_path)
    while (cap.isOpened()):
        ret, img = cap.read()
        break
    Ori_height, Ori_width, _ = img.shape
    print("height:%d, width:%d" % (Ori_height, Ori_width))
    height, width = int(Ori_height / factor_of_change), int(Ori_width / factor_of_change)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    # get the json
    video_name = video_path.split("\\")[-1]
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf8') as f:
            MASK_LIST = json.load(f).get(video_name, [])

    # MASK_LIST = mask_list_change_factor(MASK_LIST, 1 / factor_of_change)
    # img = cv2.imread(r'C:\Users\DELL\Desktop\1.jpg')

    diff_size = 0.1 * min(height, width)
    print("%2f" % diff_size)

    Original_Img = img.copy()
    cv2.namedWindow('src')
    cv2.setMouseCallback('src', on_mouse)
    cv2.imshow('src', dram_the_picture())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    json_path = "mask_points.json"
    # MASK_LIST = mask_list_change_factor(MASK_LIST, factor_of_change)
    with open(json_path, 'w', encoding='utf8') as f:
        json.dump({
            video_name: MASK_LIST
        }, f)

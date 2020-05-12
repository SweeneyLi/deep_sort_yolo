from tools.hyperlpr import *
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum


class Color(Enum):
    blue = 0
    yellow = 1
    green = 2
    no_plate = 3
    no_color = 4


def plt_show0(img):
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()


# def judge_color
class HSV(object):
    def __init__(self):
        self.blue = [(100, 124), (43, 255), (46, 255)]
        self.yellow = [(11, 34), (43, 255), (46, 255)]
        self.green = [(35, 77), (43, 255), (46, 255)]

        # self.blue = [(130, 150), (100, 220), (100, 210)]  # 141, 126, 103
        # self.yellow = [(20, 30), (180, 250), (50, 150)]  # 27, 185, 134
        # self.green = [(90, 110), (50, 190), (75, 150)]  # 96, 181, 139

        # lower_blue = np.array([100, 110, 110])
        # upper_blue = np.array([130, 255, 255])
        # lower_yellow = np.array([15, 55, 55])
        # upper_yellow = np.array([50, 255, 255])

    def isBlue(self, hsv):
        if self.blue[0][0] < hsv[0] < self.blue[0][1] and self.blue[1][0] < hsv[1] < self.blue[1][1] and self.blue[2][
            0] < hsv[2] < self.blue[2][1]:
            return True
        else:
            return False

    def isYellow(self, hsv):
        if self.yellow[0][0] < hsv[0] < self.yellow[0][1] and self.yellow[1][0] < hsv[1] < self.yellow[1][1] and \
                self.yellow[2][0] < hsv[2] < self.yellow[2][1]:
            return True
        else:
            return False

    def isGreen(self, hsv):
        if self.green[0][0] < hsv[0] < self.green[0][1] and self.green[1][0] < hsv[1] < self.green[1][1] and \
                self.green[2][0] < hsv[2] < self.green[2][1]:
            return True
        else:
            return False


def judge_color(Judge_HSV, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, _ = img.shape
    # print(img.shape)
    blue_sum, yellow_sum, green_sum = 0, 0, 0
    for i in range(h):
        for j in range(w):
            if Judge_HSV.isBlue(img[i][j]):
                blue_sum += 1
            elif Judge_HSV.isYellow(img[i][j]):
                yellow_sum += 1
            elif Judge_HSV.isGreen(img[i][j]):
                green_sum += 1
    max_sum = max(blue_sum, yellow_sum, green_sum)
    print(blue_sum, yellow_sum, green_sum)
    if max_sum < 10:
        return Color.no_color
    elif max_sum == blue_sum:
        return Color.blue
    elif max_sum == yellow_sum:
        return Color.yellow
    elif max_sum == green_sum:
        return Color.green


def judge_plate_color(image, position):
    left, top, right, bottom = position
    top = max(0, top - 10)
    bottom = min(image.shape[0], bottom + 10)
    # left += 20
    # right -= 20
    # left = max(0, left - 10)
    # right = min(image.shape[1], right + 10)

    plate_img = image[top:bottom, left:right]
    # plt_show0(plate_img)
    color = judge_color(Judge_HSV, plate_img)
    # print(color.name)

    return color


img_path = r"output\2.jpg"
img_list = glob.glob(r"D:\WorkSpaces\deep_sort_yolov3\output\color\*.jpg")
# img_list = [r"F:\Workplace\deep_sort_yolov3\output\plate\2.jpg"]
# img_list = glob.glob(r"F:\Workplace\deep_sort_yolov3\output\plate\*.jpg")
Judge_HSV = HSV()

truth_list = []
detect_list = []
for img_path in img_list:

    image = cv2.imread(img_path)
    plate_info = HyperLPR_plate_recognition(image)
    img_name = img_path.split("\\")[-1]

    truth_list.append(img_name[0])

    print(img_name, plate_info)
    if len(plate_info) == 0:
        detect_list.append(Color.no_plate.name)
        continue
    left, top, right, bottom = plate_info[0][2]
    top = max(0, top - 10)
    bottom = min(image.shape[0], bottom + 10)
    # left += 20
    # right -= 20
    # left = max(0, left - 10)
    # right = min(image.shape[1], right + 10)

    plate_img = image[top:bottom, left:right]
    # plt_show0(plate_img)

    color = judge_color(Judge_HSV, plate_img)
    print(color.name)

    detect_list.append(color.name)

print(truth_list)
print(detect_list)
for i in range(len(truth_list)):
    print(truth_list[i], detect_list[i])


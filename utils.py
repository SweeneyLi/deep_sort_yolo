from tools.hyperlpr import *
import matplotlib.pyplot as plt
from enum import Enum
import cv2


class VehicleClass(Enum):
    bus = 0
    taxi = 1
    coach = 2
    car = 3
    motor = 4
    heavy_truck = 5
    van = 6
    container_truck = 7
    car_nh = 8
    car_h = 9
    car_hc = 10


class_dict = {'bus': 0, 'taxi': 1, 'coach': 2, 'car': 3, 'motor': 4, 'heavy_truck': 5, 'van': 6, 'container_truck': 7,
              'car_hc': 8, 'car_h': 9, 'car_nh': 10}
# reverse_class = {v: k for k, v in class_dict.items()}
# =====================================================================================================================
yolo_score = 0.35
yolo_iou = 0.6

max_iou_distance = 0.7

n_init = 5
max_age = 30

max_cosine_distance = 0.4  # 余弦距离的控制阈值
nn_budget = 60  # len of feature maps
nms_max_overlap = 0.5  # 非极大抑制的阈值

max_area_ratio = 0.3

min_plate_score = 0.3

height_of_heavy_truck = 700
height_of_container_truck = 1000


# =====================================================================================================================
def print_leave_list(leave_list):
    res = []
    for i in VehicleClass:
        res.append(i.name + ":" + str(leave_list[i.value]))
    return ", ".join(res)


# def class2Id(v_class):
#     return class_dict[v_class]
#
#
# def id2Class(v_id):
#     return reverse_class[v_id]


def detect_class_by_plate(image, min_plate_score=0.3):
    '''

    :param image:
    :param min_plate_score:
    :return: plate, p_score, plate_position
    '''
    # plt.figure(figsize=[12, 8])
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()

    try:
        plate_info = HyperLPR_plate_recognition(image)
    except Exception as e:
        print(e)
        return None, 0, None

    if len(plate_info) == 0 or plate_info[0][1] < min_plate_score:
        return None, 0, None

    return plate_info[0][0], plate_info[0][1], plate_info[0][2]


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


def judge_color(image, position):
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

Judge_HSV = HSV()
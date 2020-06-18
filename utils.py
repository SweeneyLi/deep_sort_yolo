from tools.hyperlpr import *
import matplotlib.pyplot as plt
from enum import Enum
import cv2
from parameter import *
import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont

import uuid


# third data
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


class Color(Enum):
    blue = 0
    yellow = 1
    green = 2
    no_plate = 3
    no_color = 4


# class_dict = {'bus': 0, 'taxi': 1, 'coach': 2, 'car': 3, 'motor': 4, 'heavy_truck': 5, 'van': 6, 'container_truck': 7,
#               'car_hc': 8, 'car_h': 9, 'car_nh': 10}

class_dict = {'公交车': 0, '出租车': 1, '大客车': 2, '小客车': 3, '摩托车': 4, '大货车': 5, '小货车': 6, '集卡': 7,
              '非沪牌': 8, '沪牌': 9, '沪C': 10}


# reverse_class = {v: k for k, v in class_dict.items()}

# def print_leave_list(leave_list):
#     res = []
#     # for i in VehicleClass:
#     #     res.append(i.name + ":" + str(leave_list[i.value]))
#     for i, v in class_dict.items():
#         res.append(i + ":" + str(leave_list[v]))
#     return ", ".join(res)

def print_leave_list(leave_list):
    res = []
    new_list = leave_list.copy()
    class_dict2 = {'公交车': 0, '摩托车': 4, '大货车': 5, '小货车': 6, '集卡': 7,
                   '非沪牌': 8, '沪牌': 9, '沪C': 10}
    new_list[0] += new_list[2]
    new_list[8] += new_list[3]

    new_list[9] += new_list[10]
    new_list[10] = 0
    for i, v in class_dict2.items():
        res.append(i + ":" + str(new_list[v]))
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
    :return: plate, p_color, p_score
    '''

    # test
    # plt.figure(figsize=[12, 8])
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
    h, w, _ = image.shape
    # uuid_str = uuid.uuid4().hex
    # tmp_file_name = 'tmpfile_%s.jpg' % uuid_str
    # cv2.rectangle(image, (int(0), int(h / 4 * 3)), (int(w), int(h)), (1, 1, 1), 5)
    # plt.imsave(r"output/tmp/" + tmp_file_name, image)

    image = image[int(h * vehicle_plate_height_ratio):, :, :]
    try:
        plate_info = HyperLPR_plate_recognition(image)
    except Exception as e:
        print(e)
        return None, Color.no_plate.value, 0

    if len(plate_info) == 0 or plate_info[0][1] < min_plate_score:
        return None, Color.no_plate.value, 0

    p_color = judge_plate_color(image, plate_info[0][2])

    # uuid_str = uuid.uuid4().hex
    # tmp_file_name = '%s%s_%s.jpg' % (plate_info[0][0], p_color.name, uuid_str)
    # cv2.rectangle(image, (int(0), int(h / 4 * 3)), (int(w), int(h)), (1, 1, 1), 5)
    # plt.imsave(r"output/tmp/" + tmp_file_name, image)

    return plate_info[0][0], p_color, plate_info[0][1]


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
    # for the speed
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, _ = img.shape
    # print(img.shape)
    blue_sum, yellow_sum, green_sum = 0, 0, 0
    for i in range(0, h, 3):
        for j in range(0, w, 3):
            if Judge_HSV.isBlue(img[i][j]):
                blue_sum += 1
            elif Judge_HSV.isYellow(img[i][j]):
                yellow_sum += 1
            elif Judge_HSV.isGreen(img[i][j]):
                green_sum += 1
    max_sum = max(blue_sum, yellow_sum, green_sum)

    # test
    # print(blue_sum, yellow_sum, green_sum)

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

    # r, g, b = cv2.split(plate_img)
    # plate_img = cv2.merge([b, g, r])

    # test
    # plt_show0(plate_img)

    color = judge_color(Judge_HSV, plate_img)

    # write the plate of different color
    # uuid_str = uuid.uuid4().hex
    # tmp_file_name = 'tmpfile_%s.jpg' % uuid_str
    # plt.imsave("output/color/" + str(color.name) + '/' + tmp_file_name, cv2.cvtColor(plate_img, cv2.COLOR_RGB2BGR))

    return color.value


Judge_HSV = HSV()


# def judge_vehicle_type(vehicle_class, vehicle_score, height, plate, p_color):
#     """
#
#     :param vehicle_class: value
#     :param vehicle_score:
#     :param height:
#     :param plate:
#     :return: new_class, new_scores
#     """
#     # value to class
#     vehicle_class = VehicleClass(vehicle_class)
#
#     # rise the score of the rare class
#     if vehicle_class in [VehicleClass.bus, VehicleClass.coach]:
#         # vehicle_class = judge_bus_coach_by_plate(plate)
#         # vehicle_score = plate_constant_score
#         # vehicle_score += (1 - vehicle_score) / 3
#         vehicle_score = 0.999
#         vehicle_class = VehicleClass.coach
#         pass
#
#     if vehicle_class != VehicleClass.coach and vehicle_class != VehicleClass.bus and height > height_of_heavy_truck:
#         if height > height_of_container_truck:
#             vehicle_class = VehicleClass.container_truck
#             vehicle_score = height_container_truck_score
#         else:
#             vehicle_class = VehicleClass.heavy_truck
#             vehicle_score = height_heavy_truck_score
#
#     if vehicle_class in [VehicleClass.taxi, VehicleClass.car, VehicleClass.van]:
#         # vehicle_class = judge_taxi_by_plate(plate)
#         # vehicle_score = plate_constant_score
#         pass
#
#     if vehicle_class == VehicleClass.heavy_truck:
#         if p_color == PlateColor.blue:
#             vehicle_class = VehicleClass.van
#         elif p_color == PlateColor.yellow:
#             vehicle_class = VehicleClass.heavy_truck
#     elif vehicle_class in [VehicleClass.coach, VehicleClass.car, VehicleClass.van]:
#         if p_color == PlateColor.yellow and vehicle_class != VehicleClass.coach:
#             vehicle_class = VehicleClass.coach
#         elif plate:
#             if plate[0] == '沪':
#                 if plate[1].lower() == 'c':
#                     vehicle_class = VehicleClass.car_hc
#                 else:
#                     vehicle_class = VehicleClass.car_h
#             else:
#                 vehicle_class = VehicleClass.car_nh
#
#     return vehicle_class.value, vehicle_score


def judge_vehicle_type(vehicle_class_v, vehicle_score, height, plate, p_color):
    """

    :param vehicle_class: value
    :param vehicle_score:
    :param height:
    :param plate:
    :return: new_class, new_scores
    """

    # get the bus and coach
    if vehicle_class_v == VehicleClass.bus.value:
        pass
        # if not p_color == Color.yellow.value:
        #     vehicle_class_v = VehicleClass.van.value
        # vehicle_class = judge_bus_coach_by_plate(plate)
        # vehicle_score = plate_constant_score
        # vehicle_score += (1 - vehicle_score) / 3
        # vehicle_score = 0.999


    # get the container truck by height
    elif height > height_of_truck:
        if height > height_of_container_truck:
            vehicle_class_v = VehicleClass.container_truck.value
            vehicle_score = height_container_truck_score
        else:
            vehicle_class_v = VehicleClass.van.value
            vehicle_score = height_truck_score

    # get the taxi by plate
    if vehicle_class_v == VehicleClass.car.value:
        # vehicle_class = judge_taxi_by_plate(plate)
        # vehicle_score = plate_constant_score
        pass

    if vehicle_class_v == VehicleClass.van.value:
        if p_color == Color.yellow.value:
            vehicle_class_v = VehicleClass.heavy_truck.value

    elif vehicle_class_v in [VehicleClass.coach.value, VehicleClass.car.value]:
        if p_color == Color.yellow.value:
            vehicle_class_v = VehicleClass.coach.value
        elif plate:
            if plate[0] == '沪' or p_color == Color.green.value:
                if plate[1].lower() == 'c':
                    vehicle_class_v = VehicleClass.car_hc.value
                else:
                    vehicle_class_v = VehicleClass.car_h.value
            else:
                vehicle_class_v = VehicleClass.car_nh.value

    return vehicle_class_v, vehicle_score


# draw chinese in img
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

def is_same_deque(nums_deque):
    nums = nums_deque[0]
    for n in nums_deque:
        if not n == nums:
            return False
    return True
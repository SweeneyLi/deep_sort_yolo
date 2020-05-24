#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from timeit import time
import warnings
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from keras import backend
from utils import *
import csv
import glob
import numpy as np
from math import ceil

from line_profiler import LineProfiler

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

warnings.filterwarnings('ignore')
backend.clear_session()
# =====================================================================================================================


Width = None
Height = None
Half_Height = None


def main(video_path, output_path, vehicle_file_path, sum_file_path, goal):
    start = time.time()

    # https://blog.csdn.net/weixin_43249191/article/details/84072494
    # the cap
    cap = cv2.VideoCapture(video_path)
    cap.set(6, cv2.VideoWriter.fourcc('m', 'p', '4', 'v'))
    fps = ceil(cap.get(cv2.CAP_PROP_FPS))

    Interval = fps * 30

    Width, Height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Half_Height = Height / 2

    FrameNumber = cap.get(7)
    duration = FrameNumber / fps


    print("fps: %s, width: %s, height: %s" % (fps, Width, Height))


    sum_file = open(sum_file_path, 'w', encoding='gbk')
    csv_writer2 = csv.writer(sum_file)
    csv_writer2.writerow(
        ["bus", "taxi", "coach", "car", "motor", "heavy_truck", "van", "container_truck", "car_nh", "car_h", "car_hc",
         "bus", "taxi", "coach", "car", "motor", "heavy_truck", "van", "container_truck", "car_nh", "car_h", "car_hc"])


    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)


    print("load need: " + str(time.time() - start) + 's')
    print(goal + " start!")
    leave_list = [[0 for _ in range(11)], [0 for _ in range(11)]]

    start = time.time()
    skip_frame = 0
    interval_frame = Interval
    while 1:
        ret, frame = cap.read()  # frame shape 640*480*3
        if not ret:
            break
        interval_frame -= 1

        if skip_frame:
            skip_frame -= 1
            continue

        image = Image.fromarray(frame)

        # # test
        # lp = LineProfiler()
        # lp.add_function(detect_class_by_plate)
        # lp.add_function(judge_vehicle_type)
        # lp_wrapper = lp(yolo.detect_image)
        # boxs, class_names, class_scores, plate_list, p_score_list = lp_wrapper(image)
        # lp.print_stats()

        boxs, class_names, class_scores, plate_list, p_score_list = yolo.detect_image(image)

        if len(boxs) < 3:
            skip_frame = 3 - len(boxs)

        features = encoder(frame, boxs)
        detections = [Detection(bbox, feature, v_class, v_score, plate, p_score) for
                      bbox, feature, v_class, v_score, plate, p_score in
                      zip(boxs, features, class_names, class_scores, plate_list, p_score_list)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.v_score for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, max_area_ratio, scores)
        detections = [detections[i] for i in indices]


        tracker.predict()
        tracker.update(detections)

        for i in tracker.leaves:
            path = i.center_path
            direction = -1

            if len(path) > 10:
                max_y = i.to_tlbr_int()[3]
                tmp = path[0][1] - path[-1][1]
                if max_y > Half_Height and tmp < 0:
                    direction = 0
                elif max_y < Half_Height and tmp > 0:
                    direction = 1

            if direction != -1:
                leave_list[direction][i.v_class.value] += 1

        if not interval_frame:
            interval_frame = Interval
            csv_writer2.writerow(leave_list[0] + leave_list[1])
            print("in:", (leave_list[0]), "out:", (leave_list[1]))

    cap.release()
    cv2.destroyAllWindows()

    csv_writer2.writerow(leave_list[0] + leave_list[1])

    seconds = time.time() - start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    sum_file.write(goal + " run need: %02d:%02d:%02d" % (h, m, s))
    sum_file.write("1:%02d" % (seconds/ duration))
    sum_file.close()

    print(goal + " " + str(duration) + "s - Finish!")
    print("in:", (leave_list[0]), "out:", (leave_list[1]))
    print("all: in: %d, out:%d" % (sum(leave_list[0]), sum(leave_list[1])))


    print("%02d:%02d:%02d" % (h, m, s))
    print("1:%f" % (seconds/ duration))

    return leave_list, duration


# video_list = [r"F:\Workplace\yolo_data\videos\IMG_2712.MOV"]
# video_list = glob.glob(r"D:\WorkSpaces\videos\*.MOV")
video_list = [r"D:\WorkSpaces\videos\DJI_0005.MOV"]


def run():
    for video_path in video_list:
        goal = video_path.split(".")[0].split("\\")[-1]
        print(goal)
        output_path = "output/r_%s.mov" % goal
        vehicle_file = "output/vehicle_%s.csv" % goal
        sum_file = "output/num_%s.csv" % goal

        # lp = LineProfiler()
        # lp.add_function(yolo.detect_image)
        # lp_wrapper = lp(main)
        # lp_wrapper(video_path, output_path, vehicle_file, sum_file, goal)
        # lp.print_stats()

        main(video_path, output_path, vehicle_file, sum_file, goal)

    # ===================================================================
    # parameter_file = open("output/para/parameter.csv", 'w', encoding='gbk')
    # csv_writer = csv.writer(parameter_file)
    # csv_writer.writerow(['num', 'n_init', 'max_age', 'nn_budget', 'avg_time'] +
    #                     ["bus", "taxi", "coach", "car", "motor", "heavy_truck", "van", "container_truck", "car_nh",
    #                      "car_h", "car_hc",
    #                      "bus", "taxi", "coach", "car", "motor", "heavy_truck", "van", "container_truck", "car_nh",
    #                      "car_h", "car_hc"])
    #
    # n_init_list = [5, 10, 15]
    # yolo_score_list = [0.3, 0.4, 0.5]
    # model_image_sizet_list = [(640, 640), (608, 608)]
    # # height_of_heavy_truck_list = [850, 900, 950, 1000]
    # # height_of_container_truck_list = [1250, 1300]
    #
    # num = 0
    # global yolo_score
    # global model_image_size
    # global n_init
    #
    # for i in model_image_sizet_list:
    #     for j in max_age_list:
    #         for k in nn_budget_list:
    #             n_init = i
    #             max_age = j
    #             nn_budget = k
    #
    #             path = 'output/para/' + str(num)
    #             if not os.path.exists(path):
    #                 os.mkdir(path)
    #
    #             video_path = video_list[0]
    #             goal = video_path.split(".")[0].split("\\")[-1]
    #             output_path = "output/para/%s/r_%s.mov" % (num, goal)
    #             vehicle_file = "output/para/%s/vehicle_%s.csv" % (num, goal)
    #             sum_file = "output/para/%s/nums_%s.csv" % (num, goal)
    #             leave_list, time_need = main(video_path, output_path, vehicle_file, sum_file, goal)
    #             # leave_list, time_need = [[],[]], 0
    #             csv_writer.writerow(
    #                 [num, n_init, max_age, nn_budget, time_need] + leave_list[0] +
    #                 leave_list[1])
    #             num += 1
    #
    #             if os.path.exists("output/stop.txt"):
    #                 parameter_file.close()
    #                 return
    # parameter_file.close()


yolo = YOLO()
run()
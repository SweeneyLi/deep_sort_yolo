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
from collections import deque
from keras import backend
from utils import *
import csv
import glob
import imutils
import numpy as np

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
warnings.filterwarnings('ignore')
backend.clear_session()
# =====================================================================================================================


writeVideo_flag = True
show_real_time = False
cut_size = 250
# cut_size = 0

# start_frame = None
end_frame = None

start_frame = 70
# end_frame = 2 * 30

# start_frame = 227 * 30
# end_frame = 20

# =====================================================================================================================

# pts = [deque(maxlen=30) for _ in range(9999)]
pts = {}

np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

Width = None
Height = None
title_height = 182


def main(video_path, output_path, vehicle_file, sum_file, goal):
    start = time.time()
    cap = cv2.VideoCapture(video_path)
    cap.set(6, cv2.VideoWriter.fourcc('m', 'p', '4', 'v'))
    frame_index = -1

    fps = cap.get(cv2.CAP_PROP_FPS)
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - cut_size)
    Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Width, Height = Height, Width
    print("fps: %s, width: %s, height: %s" % (fps, Width, Height))

    if writeVideo_flag:
        # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(output_path, fourcc, fps,
                              (Width, title_height + Height - cut_size))

    # write in file
    vehicle_file = open(vehicle_file, 'w', encoding='gbk')
    csv_writer = csv.writer(vehicle_file)
    csv_writer.writerow(
        ['frame', 'direction', 'vehicle_id', 'vehicle_type', 'vehicle_score', 'plate', 'plate_score', 'len_of_path',
         'width', 'height', 'minX', 'minY', 'maxX', 'maxY'])

    sum_file = open(sum_file, 'w', encoding='gbk')
    csv_writer2 = csv.writer(sum_file)
    csv_writer2.writerow(
        ["bus", "taxi", "coach", "car", "motor", "heavy_truck", "van", "container_truck", "car_nh", "car_h", "car_hc",
         "bus", "taxi", "coach", "car", "motor", "heavy_truck", "van", "container_truck", "car_nh", "car_h", "car_hc"])

    # deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    fps = 0.0

    time2 = time.time()
    print("load need: " + str(time2 - start) + 's')
    print(goal + " start!")
    leave_list = [[0 for _ in range(11)], [0 for _ in range(11)]]

    start = time.time()
    while True:
        ret, frame = cap.read()  # frame shape 640*480*3
        if not ret:
            break

        frame_index += 1

        # if frame_index & 1 == 1:
        #     continue

        if start_frame and frame_index < start_frame:
            continue
        if end_frame and frame_index >= end_frame:
            break
        t1 = time.time()

        # frame = imutils.rotate(frame, -90)

        if cut_size:
            frame = frame[cut_size:, :, :]  # cut the top of image

        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs, class_names, class_scores, plate_list, p_score_list = yolo.detect_image(image)

        features = encoder(frame, boxs)
        detections = [Detection(bbox, feature, v_class, v_score, plate, p_score) for
                      bbox, feature, v_class, v_score, plate, p_score in
                      zip(boxs, features, class_names, class_scores, plate_list, p_score_list)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.v_score for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, max_area_ratio, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for i in tracker.leaves:
            path = pts[i.track_id]
            direction = -1

            # if len(path) < 10:
            #     print(frame_index, i.track_id, len(path), path[0][1] - path[-1][1])
            if len(path) > 10:
                if path[0][1] - path[-1][1] < 0 and i.to_tlbr_int()[3] > Height / 2:
                    # if path[0][1] - path[-1][1] < 0:
                    direction = 0
                elif path[0][1] - path[-1][1] > 0 and i.to_tlbr_int()[3] < Height / 2:
                    direction = 1

            if direction != -1:
                leave_list[direction][i.v_class.value] += 1
                position = i.to_tlbr_int()
                x = int(position[0])
                y = int(position[1])
                w = int(position[2] - position[0])
                h = int(position[3] - position[1])
                if x < 0:
                    w = w + x
                    x = 0
                if y < 0:
                    h = h + y
                    y = 0
                position = [w, h, x, y, x + w, y + h]

                csv_writer.writerow(
                    [frame_index, direction, i.track_id, i.v_class.name, i.v_score, i.plate, i.p_score, len(path)]
                    + position)

            del pts[i.track_id]

        i = int(0)
        indexIDs = []

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                # print("no:", track.track_id, track.v_class)
                continue
            # boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))

            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color), 3)

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]) - 50), (int(bbox[2]), int(bbox[1])), (color), cv2.FILLED)
            cv2.putText(frame, str(track.track_id) + "-" + str(track.v_class.name) + '-' + str(round(track.v_score, 2)),
                        (int(bbox[0]), int(bbox[1])), 0,
                        5e-3 * 300, (255, 255, 255), 3)

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[1]) + 50), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, str(int(bbox[2]) - int(bbox[0])) + ", " + str(int(bbox[3]) - int(bbox[1])),
                        (int(bbox[0]), int(bbox[1] + 50)), 0, 5e-3 * 300, (255, 255, 255), 3)

            i += 1
            center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            if not pts.get(track.track_id):
                pts[track.track_id] = deque(maxlen=30)
            pts[track.track_id].append(center)
            cv2.circle(frame, (center), 5, color, 5)

            # draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), (color), thickness)

        # frame = np.concatenate((np.zeros((title_height, Height, 3), dtype="uint8"), frame))
        frame = np.concatenate((np.zeros((title_height, Width, 3), dtype="uint8"), frame))
        # cv2.rectangle(frame, (0, 0), (frame.shape[1], 200), (0, 0, 0), cv2.FILLED)

        cv2.putText(frame, "frame:%d" % frame_index, (int(0), int(250)), 0, 5e-3 * 400, (0, 255, 0), 4)
        cv2.putText(frame, "FPS: %f" % (fps), (int(0), int(300)), 0, 5e-3 * 400, (0, 255, 0), 4)

        cv2.putText(frame, "in:" + str(sum(leave_list[0])), (int(0), int(80)), 0, 5e-3 * 400, (255, 123, 255), 5)
        cv2.putText(frame, "out:" + str(sum(leave_list[1])), (int(0), int(180)), 0, 5e-3 * 400, (255, 123, 255), 5)
        cv2.putText(frame, print_leave_list(leave_list[0]), (int(300), int(80)), 0, 5e-3 * 280, (255, 255, 255), 3)
        cv2.putText(frame, print_leave_list(leave_list[1]), (int(300), int(180)), 0, 5e-3 * 280, (255, 255, 255), 3)

        # show the instant result
        if show_real_time:
            cv2.namedWindow("YOLO3_Deep_SORT", 0)
            cv2.resizeWindow('YOLO3_Deep_SORT', 1024, 768)
            cv2.imshow('YOLO3_Deep_SORT', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)

        fps = (fps + (1. / (time.time() - t1))) / 2

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_index % 300 == 0:
            print("1s cost:" + str((time.time() - start) / 10))
            start = time.time()
            print(frame_index / 30, "in:", (leave_list[0]), "out:", (leave_list[1]))
            csv_writer2.writerow(leave_list[0] + leave_list[1])

        # test
        if os.path.exists("output/stop.txt"):
            break

    csv_writer2.writerow(leave_list[0] + leave_list[1])

    print(" ")
    print(goal + " " + str(frame_index) + " frame - Finish!")

    print("in:", (leave_list[0]), "out:", (leave_list[1]))
    print("all: in: %d, out:%d" % (sum(leave_list[0]), sum(leave_list[1])))
    time_need = round((time.time() - time2) / 60, 2)
    print("run need: " + str(time_need) + 'min')

    vehicle_file.write('\n')
    vehicle_file.write(goal + " run need: " + str(time_need) + 'm')
    cap.release()
    if writeVideo_flag:
        out.release()
        vehicle_file.close()
        sum_file.close()
    cv2.destroyAllWindows()

    return leave_list, time_need


# video_list = [r"F:\Workplace\yolo_data\videos\IMG_2712.MOV"]
# video_list = glob.glob(r"D:\WorkSpaces\videos\*.MOV")
video_list = [r"D:\WorkSpaces\videos\DJI_0005.MOV"]


# video_list = [r"D:\WorkSpaces\videos\60s.mp4"]

def run():
    for video_path in video_list:
        goal = video_path.split(".")[0].split("\\")[-1]
        print(goal)
        output_path = "output/r_%s.mov" % goal
        vehicle_file = "output/vehicle_%s.csv" % goal
        sum_file = "output/num_%s.csv" % goal
        main(video_path, output_path, vehicle_file, sum_file, goal)

    # ===================================================================
    # parameter_file = open("output/para/parameter.csv", 'w', encoding='gbk')
    # csv_writer = csv.writer(parameter_file)
    # csv_writer.writerow(['num', 'n_init', 'max_cosine_distance', 'height_of_heavy_truck', 'avg_time'] +
    #                     ["bus", "taxi", "coach", "car", "motor", "heavy_truck", "van", "container_truck", "car_nh",
    #                      "car_h", "car_hc",
    #                      "bus", "taxi", "coach", "car", "motor", "heavy_truck", "van", "container_truck", "car_nh",
    #                      "car_h", "car_hc"])
    #
    # n_init_list = [8]
    # max_cosine_distance_list = [0.55, 0.6, 0.65, 0.7, 0.75]  # 余弦距离的控制阈值
    # # nms_max_overlap_list = [0.50, 0.55, 0.6]  # 非极大抑制的阈值
    #
    # height_of_heavy_truck_list = [1000]
    # # height_of_heavy_truck_list = [850, 900, 950, 1000]
    # # height_of_container_truck_list = [1250, 1300]
    #
    # num = 0
    # for i in n_init_list:
    #     for j in max_cosine_distance_list:
    #         for k in height_of_heavy_truck_list:
    #             n_init = i
    #             max_cosine_distance = j
    #             height_of_heavy_truck = k
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
    #                 [num, n_init, max_cosine_distance, height_of_heavy_truck, time_need / 90] + leave_list[0] +
    #                 leave_list[1])
    #             num += 1
    #
    #             if os.path.exists("output/stop.txt"):
    #                 parameter_file.close()
    #                 return
    # parameter_file.close()


yolo = YOLO()
run()

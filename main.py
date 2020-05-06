#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from timeit import time
import cv2
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
import numpy as np
import glob

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
warnings.filterwarnings('ignore')
backend.clear_session()
# =====================================================================================================================
# video_path = "output/31s.mov"
# video_path = r"F:\Workplace\yolo_data\videos\16s.mp4"


writeVideo_flag = True
show_real_time = False
cut_size = 250

# start_frame = None
# end_frame = None

start_frame = 3 * 30
end_frame = 10 * 30
# =====================================================================================================================

# pts = [deque(maxlen=30) for _ in range(9999)]
pts = {}

np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

Width = None
Height = None
title_height = 182


def main(video_path, output_path, leave_file_name):
    start = time.time()
    cap = cv2.VideoCapture(video_path)
    cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    frame_index = -1

    fps = cap.get(cv2.CAP_PROP_FPS)
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - cut_size)
    Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("fps: %s, width: %s, height: %s" % (fps, Width, Height))

    if writeVideo_flag:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                              (Width, title_height + Height - cut_size))

    leave_file = open(leave_file_name, 'w', encoding='utf-8')
    csv_writer = csv.writer(leave_file)
    csv_writer.writerow(
        ['frame', 'direction', 'vehicle_id', 'vehicle_type', 'vehicle_score', 'plate', 'plate_score', 'len_of_path',
         'minX', 'minY', 'maxX', 'maxY'])

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
    while True:

        ret, frame = cap.read()  # frame shape 640*480*3
        if not ret:
            break

        frame_index += 1

        if start_frame and frame_index < start_frame:
            continue
        if end_frame and frame_index >= end_frame:
            break
        t1 = time.time()

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
            cv2.putText(frame, str(track.track_id) + "-" + str(track.v_class.name), (int(bbox[0]), int(bbox[1])), 0,
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

        cv2.putText(frame, "frame:%d" % (frame_index), (int(20), int(0)), 0, 5e-3 * 400, (0, 255, 0), 3)
        cv2.putText(frame, "FPS: %f" % (fps), (int(20), int(0)), 0, 5e-3 * 400, (0, 255, 0), 3)

        frame = np.concatenate((np.zeros((title_height, Width, 3), dtype="uint8"), frame))
        # cv2.rectangle(frame, (0, 0), (frame.shape[1], 200), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, "in:" + str(sum(leave_list[0])), (int(0), int(80)), 0, 5e-3 * 400, (255, 255, 255), 5)
        cv2.putText(frame, "out:" + str(sum(leave_list[1])), (int(0), int(180)), 0, 5e-3 * 400, (255, 255, 255), 5)
        cv2.putText(frame, print_leave_list(leave_list[0]), (int(180), int(80)), 0, 5e-3 * 300, (255, 0, 255), 2)
        cv2.putText(frame, print_leave_list(leave_list[1]), (int(180), int(180)), 0, 5e-3 * 300, (255, 0, 255), 2)
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

        if frame_index % 900 == 0:
            print(frame_index, "in:", (leave_list[0]), "out:", (leave_list[1]))

    print(" ")
    print(goal + " " + str(frame_index) + "frame - Finish!")
    print(frame_index)
    print("in:", (leave_list[0]), "out:", (leave_list[1]))
    end = time.time()
    print("run need: " + str((end - time2) / 60) + 'min')

    leave_file.write('\n')
    leave_file.write(goal + " run need: " + str(end - time2) + 's')
    cap.release()
    if writeVideo_flag:
        out.release()
        leave_file.close()
    cv2.destroyAllWindows()


# video_list = [r"F:\Workplace\yolo_data\videos\IMG_2712.MOV"]
video_list = glob.glob(r"F:\Workplace\yolo_data\videos\*.MOV")

if __name__ == '__main__':
    yolo = YOLO()
    for video_path in video_list:
        goal = video_path.split(".")[0].split("\\")[-1]
        print(goal)
        output_path = "output/r_%s.mov" % goal
        leave_file_name = "output/leave_%s.csv" % goal

        main(video_path, output_path, leave_file_name)

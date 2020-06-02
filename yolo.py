#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import random
import numpy as np
from keras import backend as K
# from keras.models import load_model
from tensorflow.keras.models import load_model
# import tensorflow as tf
# from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval
from yolo3.utils import letterbox_image
import argparse
from utils import *

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video", default="./test_video/det_t1_video_00315_test.avi")
ap.add_argument("-c", "--class", help="name of class", default="person")
args = vars(ap.parse_args())

model_dir = 'yolov3_640'


# model_dir = 'yolov4_ori_anchor'

class YOLO(object):
    def __init__(self):
        self.model_path = './model_data/' + model_dir + '/yolo.h5'
        self.anchors_path = 'model_data/' + model_dir + '/yolo_anchors.txt'
        self.classes_path = 'model_data/coco.names'
        # 具体参数可实验后进行调整

        self.score = yolo_score
        self.iou = yolo_iou
        self.model_image_size = model_image_size

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()

        self.plate_aero_height = g_env['input']['height'] * plate_aero_height_ratio
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=config)
        # K.set_session(self.sess)

        self.sess = K.get_session()

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=config)
        # K.set_session(self.sess)

        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        # print(class_names)
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

        self.yolo_model = load_model(model_path, compile=False)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.is_fixed_size:
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        return_boxs = []
        return_class_name = []
        return_scores = []
        return_plate = []
        return_p_scores = []

        for i, c in reversed(list(enumerate(out_classes))):

            box = out_boxes[i]
            x = int(box[1])
            y = int(box[0])
            w = int(box[3] - box[1])
            h = int(box[2] - box[0])
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h + y
                y = 0
            return_boxs.append([x, y, w, h])

            # return_boxs.append([x, y, x + w, y + h])
            # if h > 400 and y > image.size[0] / 2:

            if box[2] > self.plate_aero_height:
                plate, p_color, p_score = detect_class_by_plate(np.array(image)[y:y + h, x: x + w, :], min_plate_score)
            else:
                plate, p_color, p_score = None, None, 0
            # plate, p_color, p_score = None, None, 0
            # plate, p_color, p_score = detect_class_by_plate(np.array(image)[y:y + h, x: x + w, :], min_plate_score)

            c, out_scores[i] = judge_vehicle_type(c, out_scores[i], h, plate, p_color)
            return_plate.append(plate)
            return_p_scores.append(p_score)

            return_class_name.append(VehicleClass(c))
            return_scores.append(out_scores[i])

        return return_boxs, return_class_name, return_scores, return_plate, return_p_scores

    def detect_image2(self, image):
        # if self.is_fixed_size:
        #     assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        #     assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        #     boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        # else:
        #     new_image_size = (image.width - (image.width % 32),
        #                       image.height - (image.height % 32))
        #     boxed_image = letterbox_image(image, new_image_size)
        # image_data = np.array(boxed_image, dtype='float32')

        image_data = cv2.resize(image, (960, 544))
        image_data = image_data.astype('float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[1], image.shape[0]],
                K.learning_phase(): 0
            })
        return_boxs = []
        return_class_name = []
        return_scores = []
        return_plate = []
        return_p_scores = []

        for i, c in reversed(list(enumerate(out_classes))):

            box = out_boxes[i]
            x = int(box[1])
            y = int(box[0])
            w = int(box[3] - box[1])
            h = int(box[2] - box[0])
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h + y
                y = 0
            return_boxs.append([x, y, w, h])

            # return_boxs.append([x, y, x + w, y + h])
            # if h > 400 and y > image.size[0] / 2:
            if h > 400:
                plate, p_color, p_score = detect_class_by_plate(image[y:y + h, x: x + w, :], min_plate_score)
            else:
                plate, p_color, p_score = None, None, 0
            # plate, p_color, p_score = None, None, 0

            c, out_scores[i] = judge_vehicle_type(c, out_scores[i], h, plate, p_color)
            return_plate.append(plate)
            return_p_scores.append(p_score)

            return_class_name.append(VehicleClass(c))
            return_scores.append(out_scores[i])

        return return_boxs, return_class_name, return_scores, return_plate, return_p_scores

    def close_session(self):
        self.sess.close()

#! /usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image

ori = cv2.imread(r"D:\WorkSpaces\deep_sort_yolov3\output\plate\6.jpg")
res = 0
for i in range(100):
    s = time.time()
    # img = Image.fromarray(cv2.cvtColor(ori, cv2.COLOR_BGR2RGB))
    img = Image.fromarray(ori)
    # img = img[..., ::-1]
    e = time.time() - s

    # s2 = time.time()
    # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img2 = Image.fromarray(ori[..., ::-1])
    # e2 = time.time() - s2
    # res += e - e2

    res += e

print(res / 100)
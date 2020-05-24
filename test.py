#! /usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image

from parameter import  nn_budget

def main():
    print(nn_budget)


def my():
    global nn_budget
    nn_budget = 100
    main()
    print(nn_budget)


my()
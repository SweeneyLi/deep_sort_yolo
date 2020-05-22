from tools.hyperlpr import *
import glob
from utils import *
import cv2
import time

img_list = glob.glob(r"D:\WorkSpaces\deep_sort_yolov3\output\plate\*.jpg")

s = time.time()
for image in img_list:
    img = cv2.imread(image)
    h, _, _ = img.shape
    img = img[int(h / 4 * 3):, :, :]
    print(image.split("\\")[-1], detect_class_by_plate(img))
    # detect_class_by_plate(img)
print("cost:%s" % str(time.time() - s))

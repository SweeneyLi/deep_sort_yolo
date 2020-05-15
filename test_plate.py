from tools.hyperlpr import *
import glob
from utils import *
import cv2

img_list = glob.glob(r"C:\Users\DELL\Desktop\*.jpg")
for image in img_list:
    image = cv2.imread(image)
    print(detect_class_by_plate(image))

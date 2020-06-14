from tools.hyperlpr import *
import glob
from utils import *
import cv2
import time
from line_profiler import LineProfiler


# img_list = glob.glob(r"D:\WorkSpaces\deep_sort_yolov3\output\plate\*.jpg")
# img_list = glob.glob(r"D:\WorkSpaces\deep_sort_yolov3\output\color\*.jpg")
img_list = glob.glob(r"C:\Users\DELL\Desktop\*.png")

s = time.time()
for image in img_list:
    img = cv2.imread(image)
    # h, _, _ = img.shape
    # img = img[int(h / 4 * 3):, :, :]

    # print(image.split("\\")[-1])
    #
    # lp = LineProfiler()
    # lp.add_function(judge_plate_color)
    # lp_wrapper = lp(detect_class_by_plate)
    # print(lp_wrapper(img))
    # lp.print_stats()


    print(image.split("\\")[-1], detect_class_by_plate(img))
    # detect_class_by_plate(img)
    # break
print("cost:%s" % str(time.time() - s))

from tools.hyperlpr import *
import glob

img_list = glob.glob("*.*g")
for image in img_list:
    plate_info = HyperLPR_plate_recognition(image)
    img_name = img_path.split("\\")[-1]
import os
import glob

xml_dir = glob.glob(r"F:\Workplace\yolo_data\train_data\Fourth\some clip\clip_img\*.jpg")
for xml_path in xml_dir:
    os.rename(xml_path, xml_path.replace("_clip", ""))
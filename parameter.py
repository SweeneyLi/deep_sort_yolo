# yolo
yolo_score = 0.30
yolo_iou = 0.6
max_iou_distance = 0.7

# deep sort
n_init = 5
max_age = 30
max_cosine_distance = 0.4  # 余弦距离的控制阈值 0.5
nn_budget = 60  # len of feature maps
nms_max_overlap = 0.5  # 非极大抑制的阈值 0.3

max_area_ratio = 0.3

# vehicle class
min_plate_score = 0.3

height_of_heavy_truck = 860
height_of_container_truck = 1000

height_constant_score = 1
plate_constant_score = 0.8


# =====================================================================================================================
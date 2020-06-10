# global variable
g_env = {
    'input': {
        'height': 0,
        'width': 0
    },
}

plate_aero_height_ratio = 3 / 4         # the max height of the area which need to detect plate

# yolo
yolo_score = 0.5
yolo_iou = 0.6
model_image_size = (640, 640)
# model_image_size = (960, 544)
max_iou_distance = 0.7


speedUp = True
speedRate = 1
# deep sort
# n_init = max(3, int(10 / speedRate))
n_init = max(3, int(10 / speedRate))
max_age = max(10, int(20 / speedRate))
max_cosine_distance = 0.6 if speedUp else 0.8
nn_budget = 20


max_area_ratio = 0.4 # no max depression

# vehicle class
min_plate_score = 0.3

height_of_heavy_truck = 900
height_of_container_truck = 1350

height_heavy_truck_score = 0.98
height_container_truck_score = 1.01
plate_constant_score = 0.99


# =====================================================================================================================
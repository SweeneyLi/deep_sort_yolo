# 文件说明

## deep_sort

- 物体追踪定位算法
- track.py文件， update函数(144)
  - 对于每个追踪器的类别更新

## model_data

- 模型文件
- .names， 类别名称文件
- .h5模型文件

## output（含视频，测试用）

- 默认输出目录

## output_csv（只含csv，正式使用）

+ 输出目录

## tools

+ 辅助脚本

## yolov3, yolov4

- yolov3, yolov4 读取模型所需文件

## 杂

- .gitignore，git上传时配置文件
- requirments.txt python安装模块文件

## 重要

- debug.py; release.py 测试文件和正式使用文件，前者输出视频和调试信息，后者只输出结果
- parameter.py 参数配置文件
- utils.py 判别车牌颜色，类型等函数
- video_process.py 视频预处理，裁剪，预跳帧
- draw_mask.py 视频打框
- mask_points.json 打框后保存的“视频：点”的信息
- mult-direction-debug.py 打框后的多方向车辆检测



# 参数

## 视频预处理

- video_process.py
- 例如：2分钟视频，50帧/秒
- 加速比5， 变为10帧/秒， 那么建议fps也选10帧/s（计算时间比方便）
- 得到2分钟视频，10帧/秒（如果fps仍然为50，则变成24s 的50帧/秒视频，计算不方便）
- 如果原fps和加速比相除有小数，会有相应误差，建议去个整除的数字，实际调试，每秒10帧，5帧等等

## parameter.py

- ```
  # 选择需要车牌判断的区域和全部区域的占比，默认3/4，即只车牌检测视频下1/4的区域
  plate_aero_height_ratio = 1 / 2
  # 检测到车辆的区域中需要判别车牌的区域比（默认下方1 / 2）
  vehicle_plate_height_ratio = 1 / 2
  
  yolo_score = 0.5				# yolo准确度阈值，低于此阈值的标注舍去
  yolo_iou = 0.6					# yolo交并比阈值，任意两个框交并比大于阈值舍去 
  model_image_size = (640, 640)	# yolov3的模型对应输入图片大小，不用改
  
  max_iou_distance = 0.7			# deep sort算法中判定是否一个物体的阈值，两个对象的交并比超过阈值即匹配
  
  
  # deep sort
  n_init = 5			# 连续检测超过n_init次，才算成功匹配
  max_age = 8			# 超过max_age次检测失败，即为离开
  max_cosine_distance = 0.6 	# deep sort算法中判定是否为一个对象的余弦相似度阈值，大于阈值即匹配
  nn_budget = 20					# deep sort算法中保留一个对象特征图的长度
  
  
  max_area_ratio = 0.4 			# 非极大值抑制中，交并比超过此阈值的去除
  
  # vehicle class
  min_plate_score = 0.3			#　车牌检测置信度小于次阈值的即为检测失败
  
  height_of_heavy_truck = 900		#　大货车高度最小值
  height_of_container_truck = 1350	# 集卡高度最小值
  
  height_heavy_truck_score = 0.98		# 根据高度判定大货车后的置信度
  height_container_truck_score = 1.01	# 根据高度判定集卡后的置信度
  plate_constant_score = 0.99			# 根据车牌判定后的置信度
  
  ```
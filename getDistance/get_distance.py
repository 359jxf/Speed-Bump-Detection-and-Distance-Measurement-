import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

# 加载 YOLO 模型
model = YOLO('yolov7/runs/detect/train8/weights/best.pt')
model.classes = None  # 都识别

# 从YAML文件加载相机参数
def load_camera_params(yaml_path):
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    camera_matrix = np.array(params['camera_matrix'])
    distortion_coefficients = np.array(params['distortion_coefficients'])
    return camera_matrix, distortion_coefficients

camera_matrix, distortion_coefficient = load_camera_params('configs/intrinsic.yaml')

# 从YAML文件加载变换矩阵
def load_homography_params(yaml_path):
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    H = np.array(params['homography_matrix'])
    return H

H = load_homography_params("configs/homography.yaml")

# 获取距离
def get_distance(x, y):
    # 齐次坐标
    homogeneous_coordinate = np.array([x, y, 1])
    # 计算世界坐标
    world_point = np.dot(H, homogeneous_coordinate)
    # 归一化
    ratio = 1 / world_point[2]
    world_point *= ratio
    # 返回世界坐标的y值，单位为米
    return world_point[1] / 1000  # 将单位从毫米转换为米

def process_frame(frame):
    # step1 图像去畸变
    if camera_matrix is not None and distortion_coefficient is not None:
        # 如果提供了相机矩阵和畸变系数，进行去畸变处理
        distorted_point = np.array([[x1, y1]], dtype=np.float32).reshape(1, 1, 2)
        distorted_point = distorted_point.reshape(1, 1, 2)
        undistorted_point = cv2.undistortPoints(distorted_point, camera_matrix, distortion_coefficient, P=camera_matrix)
        undistorted_point = undistorted_point.reshape(-1, 2)
        x1, y1 = undistorted_point[0]
    else:
        # 如果没有提供去畸变参数，则直接使用原图
        x1, y1 = x.item(), y.item() + h.item() / 2

    # step2 检测减速带
    with torch.no_grad():
        outputs = model(frame)
    detection = outputs[0].boxes

    # step3 计算减速带终点到相机的距离
    if detection.xyxy is not None and len(detection.xyxy) > 0:
        x, y, w, h = detection.xywh[0]
        x1, y1 = x.item(), y.item() + h.item() / 2
        distance = get_distance(x1, y1)  # 计算距离

        # 实时显示距离
        if 0 < distance < 10:
            frame = outputs[0].plot()
            label = f"Distance: {distance:.6f}"
            cv2.putText(frame, label, (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

    return frame, distance  # 返回处理后的帧和距离

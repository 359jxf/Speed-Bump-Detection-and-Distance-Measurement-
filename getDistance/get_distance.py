import cv2
import numpy as np
import torch
from torchvision.ops import box_convert
import yaml
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yolov7')))
from detect import detect

# 加载 YOLO 模型

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
    undistorted_frame = cv2.undistort(frame, camera_matrix, distortion_coefficient)
    cv2.imwrite('D:\\vscode\Speed-Bump-Detection-and-Distance-Measurement-\\assets\\test\\temp.png', undistorted_frame)
    
    # step2 检测减速带
    detected_boxes = detect('D:\\vscode\Speed-Bump-Detection-and-Distance-Measurement-\\assets\\test\\temp.png')
    if len(detected_boxes)>0:
        detection = detected_boxes[0]
    else:
        cv2.putText(undistorted_frame, "detected nothing!", (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 2)
        return undistorted_frame, -1
    
    xyxy=torch.tensor(detection)

    # step3 计算减速带到相机的距离
    if xyxy is not None and len(xyxy) > 0:
        xywh = box_convert(xyxy, in_fmt="xyxy", out_fmt="xywh")
        x, y, w, h = xywh
        x1, y1 = x.item(), y.item() + h.item() / 2
        distance = get_distance(x1, y1)  # 计算距离
        
        # 实时显示距离
        x1, y1, x2, y2 = detection
        cv2.rectangle(undistorted_frame, (int(x1), int(y1)), (int(x2), int(y2)), (183, 6, 101), 2)  # 绘制框
        label = f"Distance: {distance:.6f}"
        cv2.putText(undistorted_frame, label, (int(x1), int(y1) - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (183,6,101), 2)

    return undistorted_frame, distance  # 返回处理后的帧和距离

import cv2
import os
import yaml
import numpy as np

# 从YAML文件加载相机参数
def load_camera_params(yaml_path):
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    camera_matrix = np.array(params['camera_matrix'])
    distortion_coefficients = np.array(params['distortion_coefficients'])
    return camera_matrix, distortion_coefficients

# 加载相机内参和畸变系数
camera_matrix, dist_coeffs = load_camera_params('configs/intrinsic.yaml')

cap = cv2.VideoCapture(1)  # 打开USB摄像头

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 视频编解码器
fps = cap.get(cv2.CAP_PROP_FPS)  # 帧数
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
out = cv2.VideoWriter('assets/datasetvideos/v9.avi', fourcc, fps, (width, height))  # 写入视频

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # 使用相机参数去畸变
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        
        # 显示去畸变后的图像
        cv2.imshow('frame', undistorted_frame)
        
        # 将去畸变后的帧写入视频
        out.write(undistorted_frame)
        
        # 按'q'键退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

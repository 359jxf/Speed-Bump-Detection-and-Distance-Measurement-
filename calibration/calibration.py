import cv2
import numpy as np
import glob
import yaml

# 读取棋盘标定参数
with open('configs/intrinsic.yaml', 'r') as f:
    calibration_config = yaml.load(f, Loader=yaml.FullLoader)

pattern_cols = calibration_config['pattern_cols']
pattern_rows = calibration_config['pattern_rows']
square_size_mm = calibration_config['square_size_mm']  # 每个小方块的实际大小（毫米）

# 创建棋盘格世界坐标系（3D坐标）
object_points = np.zeros((pattern_cols * pattern_rows, 3), np.float32)
object_points[:, :2] = np.indices((pattern_cols, pattern_rows)).T.reshape(-1, 2)
object_points *= square_size_mm  # 转换单位为毫米

# 存储所有角点和对应的物理坐标
all_object_points = []
all_image_points = []

# 加载图像
image_files = glob.glob('assets/images/*.png')

# 遍历每一张图像进行标定
for image_file in image_files:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 提高角点检测精度：使用亚像素精度
    ret, corners = cv2.findChessboardCorners(gray, (pattern_cols, pattern_rows), None)

    if ret:
        # 优化角点位置：使用亚像素精度
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                          criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # 将物理坐标和角点保存
        all_object_points.append(object_points)
        all_image_points.append(corners)

        # 在图像上画出棋盘格的角点
        cv2.drawChessboardCorners(img, (pattern_cols, pattern_rows), corners, ret)

        # 显示角点检测效果
        cv2.imshow('Chessboard corners', cv2.resize(img, (800, 400)))
        cv2.waitKey(500)  # 等待500毫秒显示检测结果
    else:
        print("No chessboard corners found in", image_file)

cv2.destroyAllWindows()

# 进行相机标定：计算相机内参矩阵 K 和畸变系数
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    all_object_points, all_image_points, gray.shape[::-1], None, None)

# 输出标定结果
if ret:
    print("Camera Calibration Successful!")
    print("Camera matrix (K):\n", mtx)
    print("Distortion coefficients:\n", dist)
    print("RMS re-projection error:", ret)
else:
    print("Camera Calibration Failed!")

# 保存标定结果到文件
calibration_data = {
    'camera_matrix': mtx.tolist(),
    'distortion_coefficients': dist.tolist(),
    'pattern_cols': pattern_cols,
    'pattern_rows': pattern_rows,
    'square_size_mm': square_size_mm
}

with open('configs/intrinsic.yaml', 'w') as f:
    yaml.dump(calibration_data, f)


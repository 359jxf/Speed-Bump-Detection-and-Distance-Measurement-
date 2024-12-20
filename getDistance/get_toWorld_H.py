import cv2
import numpy as np
import yaml  # 导入 PyYAML 库

imagePoints = [(397, 538), (292, 521), (308, 433), (379, 440), (372, 391), (458, 449),
                (512, 555)]  # pixel
worldPoints = [(0, 600), (-300, 600), (-300, 900), (0, 900), (0, 1200), (300, 900),
                (300, 600)]  # 地面，跟图片坐标一一对应。mm?

imagePoints = np.array(imagePoints, dtype=np.float32)
worldPoints = np.array(worldPoints, dtype=np.float32)

# 计算单应性矩阵
H, _ = cv2.findHomography(imagePoints, worldPoints)

# 输出单应性矩阵到终端
print("Homography matrix:")
print(H)

# 将单应性矩阵写入 YAML 文件
homography_data = {
    'homography_matrix': H.tolist()  # 将矩阵转换为列表以便保存为 YAML 格式
}

# 写入到 configs/homography.yaml 文件
output_file = 'configs/homography.yaml'
with open(output_file, 'w') as f:
    yaml.dump(homography_data, f, default_flow_style=False)

print(f"Homography matrix written to {output_file}")

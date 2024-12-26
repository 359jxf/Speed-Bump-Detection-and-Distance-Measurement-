import cv2
import numpy as np
import yaml  # 导入 PyYAML 库

# imagePoints = [[317,450],[317,424],[428,426],[207,424],[444,451],[190,449],[403,391],[235,390]] #mm
# worldPoints = [[0,1800],[0,2100],[600,2100],[-600,2100],[600,1800],[-600,1800],[600,2700],[-600,2700]]

imagePoints = [[173,310],[248,297],[205,337],[286,314],[344,374],[344,342],[338,300],[483,347]] #mm
worldPoints = [[-1395,3000],[-930,3600],[-930,2400],[-465,3000],[0,1800],[0,2400],[0,3600],[930,2400]]

# imagePoints = [[117,308],[90,368],[208,372],[239,333],[326,454],[400,313],[416,337],[445,379]] #mm
# worldPoints = [[-1800,3000],[-1200,1800],[-600,1800],[-600,2400],[0,1200],[600,1800],[600,2400],[600,3000]]

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

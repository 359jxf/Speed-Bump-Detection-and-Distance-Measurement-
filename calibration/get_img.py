import cv2
import os
import shutil

image_folder = 'assets/images'
if os.path.exists(image_folder):
    # 使用 shutil.rmtree() 删除整个文件夹及其内容
    shutil.rmtree(image_folder)

# 重新创建文件夹
os.makedirs(image_folder)

# 打开摄像头
cap = cv2.VideoCapture(1)  # 0表示使用内建摄像头，若有外接摄像头，可以改为1

# 初始化图像编号
img_id = 1

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # 显示当前帧
        cv2.imshow('Camera', frame)

        # 等待按键，按 'c' 键拍照，按 'q' 键退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # 按 'c' 键拍照
            # 图像路径命名为 1.png, 2.png, 3.png...
            img_path = f'assets/images/{img_id}.png'
            cv2.imwrite(img_path, frame)
            print(f'Image saved at {img_path}')

            # 增加图像编号
            img_id += 1
        elif key == ord('q'):  # 按 'q' 键退出
            break
    else:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
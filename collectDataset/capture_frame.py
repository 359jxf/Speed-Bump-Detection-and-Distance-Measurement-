import cv2
import os

def Capture(folder: str, num: int, cap: cv2.VideoCapture) -> None:

    count = 0   # nums of photos captured this time
    images_count = len(os.listdir(folder))  # num of current images
    id = images_count  # don't +1 because id starts from 0

    while True:
        success, frame = cap.read()
        if (num != -1) and (not success or count >= num):
            break

        # 显示视频流，按需可以去掉这行
        cv2.imshow('Press q to exit..', frame)

        # 自动保存每一帧图像
        img_path = f'{folder}/{id}.png'  # save into the folder
        cv2.imwrite(img_path, frame)
        print(f'Captured at {img_path}')
        id += 1
        count += 1

        # 按 'q' 键退出
        key = cv2.waitKey(1) & 0xff
        if key == ord('q') or key == ord('Q'):
            break

if __name__ == '__main__':
    cap = cv2.VideoCapture("assets/datasetvideos/v8.avi")
    Capture("./assets/dataimages", -1, cap)

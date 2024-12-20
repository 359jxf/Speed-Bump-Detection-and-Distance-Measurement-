import cv2
import os

# 用于记录点击的点
points = []

def click_corner(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 将点击的点添加到points列表
        points.append((x, y))
        # 在图像上绘制该点
        coordinate = "%d,%d" % (x, y)
        cv2.circle(image, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(image, coordinate, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0,255,0), thickness=1)


if __name__ == '__main__':
    file = "assets/images/1.png" 

    # 检查文件是否存在
    if not os.path.exists(file):
        print(f"Error: The file at path '{file}' does not exist.")
        exit()

    image = cv2.imread(file)

    # 检查图像是否成功加载
    if image is None:
        print(f"Error: Failed to load image from '{file}'. Check the file path or integrity.")
        exit()
        
    cv2.destroyAllWindows()

    cv2.namedWindow("choosePoint")
    cv2.setMouseCallback("choosePoint", click_corner)

    while True:
        cv2.imshow("choosePoint", image)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q') or key == ord('Q'):
            # 按下'q'时退出打印所有点击的点坐标
            print("Selected points:")
            for point in points:
                print(f"({point[0]}, {point[1]})")
            break

    cv2.destroyAllWindows()

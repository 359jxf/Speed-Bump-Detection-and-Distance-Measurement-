import cv2
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '', 'yolov7')))
from detect import detect

def image_measurement(image_path):
    detect(image_path)

def video_measurement(video_path):
    detect(video_path)

def camera_measurement(camera_index=0):
    detect(str(camera_index))

def main(options, path):
    # 根据 options 选择执行的操作
    if options == 'image':
        if path:
            image_measurement(path)
        else:
            print("Image path is required for image measurement.")
    elif options == 'video':
        if path:
            video_measurement(path)
        else:
            print("Video path is required for video measurement.")
    elif options == 'camera':
        camera_index = int(path) if path else 0  
        camera_measurement(camera_index)
    else:
        print("Invalid options. Choose 'image', 'video', or 'camera'.")

if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Object detection program")
    
    # 添加命令行参数
    parser.add_argument('--option', type=str, help="You can choose an option from processing image, video or camera realtime detection. ", required=True)
    parser.add_argument('--path', type=str, help="The path to the image or video file, or the index of camera.", required=True)

    # 解析命令行参数
    args = parser.parse_args()
    
    # 获取命令行传递的参数
    if args.option and args.path:
        options = args.option
        path = args.path
    else:
        print("Please provide both --option and --path arguments.")
    
    # 调用主程序
    main(options, path)

import cv2
import argparse
from getDistance.get_distance import process_frame

def image_measurement(image_path):
    image = cv2.imread(image_path)
    processed_frame, distance = process_frame(image)
    cv2.imshow("Processed Image", processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Distance: {distance:.6f}")

def video_measurement(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('result_video.avi', fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, distance = process_frame(frame)

        # 显示距离
        print(f"Distance: {distance:.6f}")
        out.write(processed_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def camera_measurement(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, distance = process_frame(frame)

        # 显示距离
        print(f"Distance: {distance:.6f}")
        cv2.imshow("Real-time Measurement", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

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
        # 如果是摄像头模式，path 可以是 None 或者传递摄像头索引
        camera_index = int(path) if path else 0  # 默认使用摄像头 0
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

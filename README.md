# Speed-Bump-Detection-and-Distance-Measurement
CV-2024fall-finalproject

## 架构

```
|--assets: 
| |--images存标定的图片
| |--datasetvideos减速带数据集视频
| |--test待测试的图像或视频
| |--results检测结果
|
|--collectDataset:
| |--record_video.py拍摄视频并保存在assets/datasetvideos
| |--capture_frame.py从已拍摄的视频中获得每一帧图像并存在yoloDetection/datasets/images
|
|--configs: 
| |--intrinsic.yaml存相机内参
| |--homography.yaml到世界坐标系的旋转矩阵
|
|--calibration:
| |--get_cali_img.py拍摄标定图像并保存在assets/images
| |--calibration.py标定并把得到的内参保存在intrinsic.yaml
| 
|--getDistance:
| |--get_toWorld_H.py定义世界坐标系并把结果保存在homography.yaml
| |--get_distance.py图像测距：step1去畸变，step2检测，step3测距 
|
|--utils:
| |--get_point_fromImg.py 获取图像上某点的像素坐标，和get_toWorld_H搭配使用
|
|--yolov7:
| |--data数据集
| |--runs
| | |--train训练得到的权重文件
| |--weights预训练的权重文件
| |--detect.py检测减速带，返回锚框坐标（cpu）
|
|--detector.py最终运行，通过命令行参数选择图像检测、视频检测或摄像头实时检测
|--birdeye.py鸟瞰图
```

## 使用方法

1. 安装相关依赖项，可以用虚拟环境。具体的遇到哪个报错就下载哪个吧

2. 更换绝对路径：`get_distance.py`五十行左右有两个，`detect.py`的18行权重文件，都改成你的路径

3. 创建assets/test存放待测试的图片或视频

4. 项目根路径下运行：

   ```python
   python detector.py --option image --path assets/test/image.png  
   # or
   python detector.py --option video --path assets/test/video.mp4  
   # or
   python detector.py --option camera --path 1 # 1是摄像头索引，这方式很卡，不建议
   ```

5. 然后就能在assets/results里看到结果，由于畸变系数图像会扭曲

6. 再测试时记得在`detector.py`更改图像或视频结果路径，否则会覆盖

   ```python
   cv2.imwrite('assets\\results\\p1.png',processed_frame)
   # or
   out = cv2.VideoWriter('assets\\results\\v1.avi', fourcc, fps, (frame_width, frame_height))
   ```

## 问题

1. 这个模型对清晰度有要求，太模糊了识别不了。

2. 现在视频和摄像头实时的方式特别慢，因为我现在处理的是每次把视频或摄像头的一帧传给yolo的detect，传递的时间耗费较多（我每次先把一帧去畸变了再传给yolo检测，再把结果返回计算得到距离）。但yolo自带视频流的检测较快，可以看看源码试试改一下。

3. 变换矩阵还未定义，没有实测过距离正确率。
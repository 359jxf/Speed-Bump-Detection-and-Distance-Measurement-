# Speed-Bump-Detection-and-Distance-Measurement
CV-2024fall-finalproject

### 架构

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
|
|--utils:
| |--get_point_fromImg.py 获取图像上某点的像素坐标，和get_toWorld_H搭配使用
| |--获取去畸变的图像并保存到assets/test
|
|--yolov7:
| |--data数据集
| |--runs
| | |--detect检测到的结果
| | |--train训练得到的权重文件
| |--weights预训练的权重文件
| |--detect.py图像测距：step1去畸变，step2检测，step3测距 
|
|--detector.py最终运行，通过命令行参数选择图像检测、视频检测或摄像头实时检测
```

### 使用方法

1. 安装相关依赖项，可以用虚拟环境。

2. 更换绝对路径：`detect.py`的49行权重文件，都改成你的路径

3. 创建assets/test存放待测试的图片或视频(去畸变的)

4. 项目根路径下运行：

   ```python
   python detector.py --option image --path assets/test/image.png  
   # or
   python detector.py --option video --path assets/test/video.mp4  
   # or
   python detector.py --option camera --path 1 # 1是摄像头索引
   ```

5. 然后就能在yolov7/runs/detect里看到结果

### 实现

1. 支持摄像头、视频、图像三种检测方式，摄像头和视频采用yolov7的loadsream类实现。

2. 可以通过更换class筛选检测：

   ```
   # class names
   names: [ 'person','speedbump' ]
   ```


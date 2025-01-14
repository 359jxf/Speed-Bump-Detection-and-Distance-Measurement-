## 标定

### 内参

标定板，获得内参，可以去畸变

### 旋转矩阵（外参）

图像和世界坐标系的转换

## yolo识别

### 收集数据

`collectDataset`文件夹下

感觉把`record_video.py`写完就可以去实验室拍摄视频了，后续回来处理

### 标注数据

用**labelimg**来标注：[LabelImg（目标检测标注工具）的安装与使用教程-CSDN博客](https://blog.csdn.net/knighthood2001/article/details/125883343)

建议标注写成英文speedbump

教训：一直闪退

- 使用了anaconda的base环境（python>3.10)，安装labelimg必须低于3.10（没用）
- python版本过高，不发生自动类型转换，需要手动转换。我把报错里提到的函数全部手动转int了，有用

为了更快标注，可以在labelimg下载目录`D:\APP\ANACONDA\Lib\site-packages\labelImg`里增加data/predefined_classes.txt来存会用到的标签。然后标注了就可以看到默认标签：

![image-20241210230628354](assets-record\image-20241210230628354.png)

此时datasets/labels下就可以看到自动生成的classes.txt，和predefined_classes.txt一样。

datasets/labels的具体的标注文件中每一行表示一个目标，以空格进行区分，分别表示目标的类别id，归一化处理之后的中心点x坐标、y坐标、目标框的w和h

### 训练数据

[Yolov7训练自己的数据集（超详细） - 玻璃公主 - 博客园](https://www.cnblogs.com/boligongzhu/p/16718242.html)

我无cuda，在cpu上跑很慢，在autodl算力云上重跑了。（用cpu训练慢，而且需要改loss.py,detect.py,train.py里面把cuda改成cpu）

jupyter拖动大文件会有加载缓冲条，没加载完不能unzip

0. 安装requirements里的东西，但是网上说里面pytorch有点问题要单独安装，不知道cuda和pytorch版本是不是要对应上

1. yolov7\data\speedbump：

![image-20241215224015052](assets-record\\image-20241215224015052.png)

先放入原始数据，然后运行yolov7\data下的data.py生成datasets里的训练集、测试集、验证集

2. 运行yolov7\train.py:

![image-20250114173744304](assets-record\image-20250114173744304.png)

3. 运行yolov7\detect.py:

![image-20250114173820595](assets-record\image-20250114173820595.png)

我还改了agnostic-nms、conf-thres、iou-thres因为测试不出框，降低要求了才输出了一些

自己标注的数据集测出来很差，因为将摄像头放在小车上视野受限，摄像头还很模糊。

### 重新训练

还得是张林老师数据集：

![image-20241221163707154](assets-record\image-20241221163707154.png)

注意每次重新训练都要删除dataset的cache

觉得训练完的效果不好，把上次的 ***last.pt*** 权重文件作为预训练权重输入到接下来要训练的网络中。我不是断点续训，如果是中途断了，把resume改成true就行（之后记得改回去）

### 模型分析

这张混淆矩阵看到容易把背景误识别。目前不知道有什么办法

![image-20241224140559998](assets-record\image-20241224140559998.png)

## 目标检测

ultralytics不支持yolov7，所以只能用detect.py检测，把检测到的图像框坐标传递给getdistance。

为了减少麻烦我去掉了detect的命令行参数，里面包含的路径还是绝对路径

注意ModuleNotFoundError报错，因为import的路径要从头文件夹开始，所以utils和models的前面都加一下`yolov7.`还是不能啊torch会报错，只能切到yolov7目录下运行。为了能通过detector调用yolo，需要在get_distance里面添加：

```
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yolov7')))
from detect import detect
```

为了通过命令行参数传源文件，需要再调整一下detect函数的传参。由于我的处理步骤是先去畸变了再调用detect检测减速带，而去畸变需要的参数是图像，detect里面的source又是图像路径，他们之间有个转换，我就先把去畸变图像存到某个临时路径里，再把这个路径传给detect。

由于yolo的detect每次调用都会产生一个新的文件夹保存处理后的图像，我就禁止了这部分代码，改为在detector里保存到同一个assets/result文件夹。但是每次要修改保存路径名为v1、p1自增。

今天我发现detect里有直接的webcam可以用，实时非常方便，可以支持视频和实时流畅进行。这样还也不用存一个临时路径了，检测速度很快。最后还是把检测三步骤全放在detect里了，有点屎山。畸变还去不了，只能假装输入前就是畸变处理过的的。时间有点急不知道咋加。
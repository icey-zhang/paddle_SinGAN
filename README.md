## 一、简介
本项目采用百度飞桨框架paddlepaddle复现：SinGAN: Learning a Generative Model from a Single Natural Image, by Jiaqing Zhang and Kai jiang (张佳青&蒋恺)


paper：[SinGAN: Learning a Generative Model from a Single Natural Image](https://paperswithcode.com/paper/singan-learning-a-generative-model-from-a)
code：[SinGAN](https://github.com/tamarott/SinGAN)

本代码包含了原论文的默认配置下的训练和测试代码。

## 二、复现结果

![Generated Results](https://github.com/icey-zhang/paddle_SinGAN/blob/main/Output/result-Paddle.png)

## 三、环境依赖

```
python -m pip install -r requirements.txt
```

此代码在python 3.7中进行了测试

## 四、实现

### 训练

要根据您自己的图像训练SinGAN模型，请将愿望训练图像放在Input/Images下，然后运行  
 
```
python train.py --input_name colusseum.png  
``` 
 
input_name 输入图像的名字

这也将使用得到的训练模型从最粗的尺度(n=0)开始生成随机样本。  

要在cpu机器上运行这段代码，在调用' main_train.py '时指定'not_cuda '


### 测试

#### 随机样本 

为了从任何初始生成规模生成随机样本，请首先训练SinGAN模型为所需图像(如上所述)，然后运行  
 
```
python test.py --input_name colusseum.png --mode random_samples
```
 
input_name 输入图像的名字

注意:为了使用完整模型，指定生成开始比例为0，从第二个比例开始生成，指定它为1，以此类推。 

 
#### 任意大小的随机样本

要生成任意大小的随机样本，请首先训练SinGAN模型为所需图像(如上所述)，然后运行  
 
```
python test.py --input_name colusseum.png --mode random_samples_arbitrary_sizes --scale_h 2 --scale_v 1
```

scale_h和scale_v调整图像的缩放比例

input_name 输入图像的名字


## 五、代码结构


```
├── TrainedModels  # 存放模型文件的路径
├── Input  # 存放数据集的路径
├── Output  # 存放程序输出的路径
    ├── log.txt #日志文件
├── SinGAN  # 定义模型，工具等
├── test.py  # 评估程序
├── README.md
├── train.py  # 训练程序
├── config.py #定义一些参数
├── requirements.txt #所需环境

```

## 六、模型信息

|  信息   |  说明 |
|  ----  |  ----  |
| 作者 | 张佳青&蒋恺 |
| 时间 | 2021.08 |
| 框架版本 | Paddle 2.1.2 |
| 应用场景 | 图像生成 |
| 下载链接 | [预训练模型](https://pan.baidu.com/s/1MGA0GT1jkgAvd0REjN1aRg) 提取码：ipbt |
| 飞桨项目 | [欢迎fork](https://aistudio.baidu.com/aistudio/projectdetail/2285122?shared=1) |

# SinGAN

[English](./README.md) | 简体中文
   
   * [SinGAN](#SinGAN)
      * [一、简介](#一简介)
      * [二、复现结果](#二复现结果)
      * [三、数据集](#三数据集)
      * [四、环境依赖](#四环境依赖)
      * [五、快速开始](#五快速开始)
         * [step1: clone](#step1-clone)
         * [step2: 训练](#step2-训练)
         * [step3: 测试](#step3-测试)
      * [六、代码结构与详细说明](#六代码结构与详细说明)
         * [6.1 代码结构](#61-代码结构)
         * [6.2 参数说明](#62-参数说明)
         *  [6.3 模型权重介绍](#63-模型权重介绍)
      * [七、模型信息](#七模型信息)

## 一、简介
本项目基于paddlepaddle框架复现SinGAN，SinGAN是一种新的可以从单个自然图像中学习的无条件生成模型。该模型包含一个全卷积GANs的金字塔结构，每个GANs负责学习不同在不同比例的图像上的patch分布。这允许生成任意大小和纵横比的新样本，具有显著的可变性，但同时保持训练图像的全局结构和精细纹理。与以往单一图像生成方案相比，该方法不局限于纹理图像，也没有条件(即从噪声中生成样本)。


**论文:**
- [1] Shaham T R, Dekel T, Michaeli T. Singan: Learning a generative model from a single natural image[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 4570-4580.<br>

**参考项目：**
- [https://github.com/tamarott/SinGAN](https://github.com/tamarott/SinGAN)

**项目aistudio地址：**
- notebook任务：[https://aistudio.baidu.com/aistudio/projectdetail/2298194](https://aistudio.baidu.com/aistudio/projectdetail/2298194)
- 脚本任务：[https://aistudio.baidu.com/aistudio/projectdetail/2298194](https://aistudio.baidu.com/aistudio/projectdetail/2298194)

## 二、复现结果

![Generated Results](https://github.com/icey-zhang/paddle_SinGAN/blob/main/Output/result-Paddle.png)

**模型下载**
模型地址：([百度云盘](https://pan.baidu.com/s/1MGA0GT1jkgAvd0REjN1aRg) 提取码：ipbt)

## 三、数据集

任意一张图片都可以作为训练集，本项目提供了一些可供训练的图片。
[Images](https://github.com/icey-zhang/paddle_SinGAN/tree/main/Input/Images)

## 四、环境依赖

- 硬件：GPU、CPU

- 框架：
  - PaddlePaddle >= 2.0.0

## 五、快速开始

### step1: clone 

```bash
# clone this repo
git clone https://github.com/icey-zhang/paddle_SinGAN.git
cd paddle_SinGAN-main
```
**安装依赖**
```bash
sh init.sh
```

### step2: 训练
- 单卡训练
```bash
python train.py --input_name colusseum.png 
```

- 分布式训练并使用多卡：
```bash
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3' train.py --input_name colusseum.png
```

### step3: 测试
- 随机样本
```bash
python test.py --input_name colusseum.png --mode random_samples --gen_start_scale 0
```
注意:为了使用完整模型，指定金字塔结构生成开始层（gen_start_scale）为0，从第二层开始生成，指定它为1，以此类推。

- 任意大小的随机样本
```bash
python test.py --input_name colusseum.png --mode random_samples_arbitrary_sizes --scale_h 2 --scale_v 1
```
- 使用预训练权重测试

下载预训练权重放在主目录下，再运行上述指令，注意权重放置位置，详细信息可见[代码结构](#61-代码结构)和[模型权重](#63-模型权重介绍)介绍
## 六、代码结构与详细说明

### 6.1 代码结构

```
./paddle_SinGAN-main
├─TrainedModels                   # 模型存放位置
├─Input                           # 存放数据集的路径
├─Output                          # 存放生成图像的位置
├─SinGAN                          # 定义模型，工具等
│  run.sh                         # 运行脚本
│  test.py                        # 评估
│  init.sh                        # 安装依赖
|  README_cn.md                   # 中文用户手册
|  README.md                      # 英文用户手
│  requirement.txt                # 依赖
│  train.py                       # 训练
│  config.py                      #定义一些模型与训练相关参数
```

### 6.2 参数说明

可以在 `train.py` 中设置训练相关参数，具体如下：

|  参数   | 默认值  | 说明 |
|  ----  |  ----  |  ----  |
| input_dir| Input/Images, 必选| 输入图片的路径 |
| input_name| None, 必选 | 输入图片的名字 |
| mode| train, 必选 | 训练与测试的模式 |

可以在 `test.py` 中设置测试相关参数，具体如下：

|  参数   | 默认值  | 说明 |
|  ----  |  ----  |  ----  |
| input_dir| Input/Images, 必选| 输入图片的路径 |
| input_name| None, 必选 | 图片的名字 |
| mode| None, 必选 | random_samples 随机生成与原图片一样大的图片；random_samples_arbitrary_sizes 随机生成任意尺寸的图片|
| gen_start_scale| None, 必选 | 金字塔结构开始的位置|
| scale_h| None, 可选 | 调整图像的缩放比例|
| scale_v| None, 可选 | 调整图像的缩放比例|

### 6.3 模型权重介绍

- 权重下载：
百度云盘：[预训练模型](https://pan.baidu.com/s/1MGA0GT1jkgAvd0REjN1aRg) 提取码：ipbt

- 权重信息
```
./paddle_SinGAN-main
├─TrainedModels                      #模型存放位置
  ├─colusseum                        #图片名字
    ├─scale_factor=0.750000,alpha=10 #反映金字塔尺寸变化比例
      ├─0                            #金字塔结构层（第0层）
        ├─netD.pdparams              #判别器的权重
        ├─netG.pdparams              #生成器的权重
        ├─z_opt.paparams             #用于训练的随机噪声
        ├─real_scale.png             #原始图片下采
        ├─G(z_opt).png               #用z_opt噪声生成的图像
        ├─fake_samples.png           #用另一任意随机噪声生成的图像
```

## 七、模型信息

关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| --- | --- |
| 发布者 | 张佳青、蒋恺|
| 时间 | 2021.08 |
| 框架版本 | Paddle 2.0.2 |
| 应用场景 | 图像生成 |
| 支持硬件 | GPU、CPU |
| 下载链接 | [预训练模型](https://pan.baidu.com/s/1MGA0GT1jkgAvd0REjN1aRg) 提取码：ipbt  |
| 在线运行 | [botebook](https://aistudio.baidu.com/aistudio/projectdetail/2298194)、[脚本任务](https://aistudio.baidu.com/aistudio/projectdetail/2298194)|

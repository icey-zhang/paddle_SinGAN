# SinGAN

English | [简体中文](./README_cn.md)
   * [SinGAN](#SinGAN)
      * [1 Introduction](#1-Introduction)
      * [2 Result](#2-Result)
      * [3 Dataset](#3-Dataset)
      * [4 Environment](#4-Environment)
      * [5 Quick start](#5-Quickstart)
         * [step1: clone](#step1-clone)
         * [step2: train](#step2-train)
         * [step3: test](#step3-test)
      * [6 Code structure](#6-Codestructure)
         * [6.1 structure](#61-structure)
         * [6.2 Parameter description](#62-Parameter-description)
         * [6.3 Model Weight](#63-Model-Weight)
      * [7 Model information](#7-Model-information)


## 1 Introduction
This project is based on the Paddlepaddle framework to reproduce SinGAN, a new unconditional generation model that can be learned from a single natural image. This model contains a fully convolution pyramid structure of GANs, and each GANs is responsible for learning different patch distributions on images of different proportions. This allows for the generation of new samples of arbitrary size and aspect ratio, with significant variability, while maintaining the global structure and fine texture of the training image. Compared with previous single image generation schemes, this method is not limited to texture images and has no conditions (that is, to generate samples from noise).


**Paper:**
- [1] Shaham T R, Dekel T, Michaeli T. Singan: Learning a generative model from a single natural image[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 4570-4580.<br>

**Reference project：**
- [https://github.com/tamarott/SinGAN](https://github.com/tamarott/SinGAN)

**The link of aistudio：**
- notebook：[https://aistudio.baidu.com/aistudio/projectdetail/2298194](https://aistudio.baidu.com/aistudio/projectdetail/2298194)
- Scrip：[https://aistudio.baidu.com/aistudio/projectdetail/2298194](https://aistudio.baidu.com/aistudio/projectdetail/2298194)

## 2 Result

![Generated Results](https://github.com/icey-zhang/paddle_SinGAN/blob/main/Output/result-Paddle.png)

**Model Weight Download**
Address：([Baiduyun](https://pan.baidu.com/s/1MGA0GT1jkgAvd0REjN1aRg) code：ipbt)

## 3 Dataset

Any image can be used as a training set, and this project provides some images to train with.
[Images](https://github.com/icey-zhang/paddle_SinGAN/tree/main/Input)

## 4 Environment

- Hardware: GPU, CPU

- Framework:
  - PaddlePaddle >= 2.0.0

## 5 Quick start

### step1: clone 

```bash
# clone this repo
git clone https://github.com/icey-zhang/paddle_SinGAN.git
cd paddle_SinGAN-main
```
**Installation dependency**
```bash
sh init.sh
```

### step2: train
-  single machine training
```bash
python train.py --input_name colusseum.png 
```

- train distributed and use multi machine training：
```bash
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3' train.py --input_name colusseum.png
```

### step3: test
- random_samples
```bash
python eval.py --input_name colusseum.png --mode random_samples --gen_start_scale 0
```
Note: To use the full model, specify that the pyramid structure generation starts at level 0 (gen_start_scale), starts at level 2, specifies that it is 1, and so on.

- random samples in arbitrary sizes
```bash
python eval.py --input_name colusseum.png --mode random_samples_arbitrary_sizes --scale_h 2 --scale_v 1
```
- test with pre-training weight

Download the pre-training weights in the main directory, then run the above command, notice where the weights are placed, detailed information can be seen in [code structure](#61-structure) and [Pre training model](#63-Pre training model)

## 6 Code structure

### 6.1 structure

```
./paddle_SinGAN-main
├─TrainedModels             
├─config                        
├─dataset                     
├─models    
├─Input
├─Output
│  run.sh                  
│  eval.py                    
│  init.sh                     
|  README_cn.md                 
|  README.md                  
│  requirement.txt               
│  train.py                      
```

### 6.2 Parameter description

Parameters related to training  can be set in `train.py`, as follows:

|  Parameters   | default  | description |
|  ----  |  ----  |  ----  |
| input_dir| Input/Images, Mandatory| The path to the image |
| input_name| None, Mandatory | The name of the image  |
| mode| train, Mandatory | Mode |

Parameters related to test  can be set in `test.py` , as follows:
|  Parameters   | default  | description |
|  ----  |  ----  |  ----  |
| input_dir| Input/Images, Mandatory| The path to the image |
| input_name| None, Mandatory | The name of the image |
| mode| None, Mandatory | Random samples or random samples with arbitrary sizes|
| gen_start_scale| None, Mandatory | The beginning of the pyramid|
| scale_h| None, Optional | Adjust the scale of the image|
| scale_v| None, Optional | Adjust the scale of the image|

### 6.3 Model Weight

- Model Weight Download：
Baiduyun：[Pre training model](https://pan.baidu.com/s/1MGA0GT1jkgAvd0REjN1aRg) 提取码：ipbt

- Model Weight Information
```
./paddle_SinGAN-main
├─TrainedModels                      #Model storage location
  ├─colusseum                        #Image name
    ├─scale_factor=0.750000,alpha=10 #Reflect the scale of pyramid size change
      ├─0                            #Pyramid structure layer (layer 0)）
        ├─netD.pdparams              #The weight of the discriminator
        ├─netG.pdparams              #The weight of the generator
        ├─z_opt.paparams             #-   Random noise for training
        ├─real_scale.png             #Original picture downsampling
        ├─G(z_opt).png               #Image generated with z opt noise
        ├─fake_samples.png           #Image generated from another arbitrary random noise
```

## 7 Model information

For other information about the model, please refer to the following table:
| information | description |
| --- | --- |
| Author | Jiaqing Zhang、Kai jiang|
| Date | 2021.08 |
| Framework version | Paddle 2.1.2 |
| Application scenarios | Image Generation |
| Support hardware | GPU、CPU |
| Download link | [Pre training model](https://pan.baidu.com/s/1MGA0GT1jkgAvd0REjN1aRg) code：ipbt  |
| Online operation | [botebook](https://aistudio.baidu.com/aistudio/projectdetail/2298194)、[Script](https://aistudio.baidu.com/aistudio/projectdetail/2298194)|

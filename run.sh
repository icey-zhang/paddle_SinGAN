### 训练
# 要根据您自己的图像训练SinGAN模型，请将期望训练图像放在Input/Images下，
# input_name: 输入图像文件名
input_name='colusseum.png'
# 这也将使用得到的训练模型从最粗的尺度(n=0)开始生成随机样本。  
echo ${input_name}
echo "start training..."
python train.py --input_name ${input_name} 
# 在cpu机器上运行:
echo "start training with cpu..."
python train.py --input_name ${input_name}  --not_cuda

### 测试
#### 生成与训练样本相同尺寸的随机新样本:
# input_name: 输入图像文件名
# mode: 
#       random_samples: 生成与训练样本相同尺寸的随机新样本
#       random_samples_arbitrary_sizes: 生成任意大小的随机样本
# gen_start_scale: 初始化生成尺度
echo "start testing..."
echo "gererating random samples with original size..."
python test.py --input_name ${input_name} --mode random_samples --gen_start_scale=0
# 在cpu机器上运行:
echo "gererating random samples with original size with cpu..."
python test.py --input_name ${input_name} --mode random_samples --gen_start_scale=0 --not_cuda

 #### 生成任意大小的随机样本
# scale_h和scale_v: 调整图像的横纵缩放比例
# input_name: 输入图像文件名
# mode: 
#       random_samples: 生成与训练样本相同尺寸的随机新样本
#       random_samples_arbitrary_sizes: 生成任意大小的随机样本
# gen_start_scale: 初始化生成尺度
echo "gererating random samples with arbitrary size..."
python eval.py --input_name ${input_name} --mode random_samples_arbitrary_sizes --scale_h 2 --scale_v 1 --gen_start_scale=0
# 在cpu机器上运行:
echo "gererating random samples with arbitrary size with cpu..."
python eval.py --input_name ${input_name} --mode random_samples_arbitrary_sizes --scale_h 2 --scale_v 1 --gen_start_scale=0 --not_cuda

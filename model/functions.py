#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import paddle.nn as nn
import scipy.io as sio
import math
from skimage import io as img
from skimage import color, morphology, filters
#from skimage import morphology
#from skimage import filters
from model.imresize import imresize
import os
import random
from sklearn.cluster import KMeans


# custom weights initialization called on netG and netD

def read_image(opt):
    """
    读取图像
    """
    x = img.imread('%s%s' % (opt.input_img,opt.ref_image))
    return np2paddle(x)

def denorm(x):
    """
    反归一化
    """
    out = (x + 1) / 2
    return out.clip(0, 1)

def norm(x):
    """
    将输入归一化到 [-1,1] 之间
    """
    out = (x -0.5) *2
    return out.clip(-1, 1)

def convert_image_np(inp):
    """
    将输入转换为np.array()    
    """
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))
    inp = np.clip(inp,0,1)
    return inp

def save_image(real_cpu,receptive_feild,ncs,epoch_num,file_name):
    """
    保存图像
    """
    fig,ax = plt.subplots(1)
    if ncs==1:
        ax.imshow(real_cpu.reshape(real_cpu.size(2),real_cpu.size(3)),cmap='gray')
    else:
        #ax.imshow(convert_image_np(real_cpu[0,:,:,:].cpu()))
        ax.imshow(convert_image_np(real_cpu.cpu()))
    rect = patches.Rectangle((0,0),receptive_feild,receptive_feild,linewidth=5,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(file_name)
    plt.close(fig)

def convert_image_np_2d(inp):
    """
    将 2d 输入转换为 2d numpy.array()    
    """
    inp = denorm(inp)
    inp = inp.numpy()
    return inp

def generate_noise(size,num_samp=1,type='gaussian', scale=1):
    """
    生成三类噪声: gaussian, gaussian_mixture, uniform
    """
    if type == 'gaussian':
        noise = paddle.randn([num_samp, size[0], round(size[1]/scale), round(size[2]/scale)],'float32')
        noise = upsampling(noise,size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = paddle.randn([num_samp, size[0], size[1], size[2]],'float32')+5
        noise2 = paddle.randn([num_samp, size[0], size[1], size[2]],'float32')
        noise = noise1+noise2
    if type == 'uniform':
        noise = paddle.randn([num_samp, size[0], size[1], size[2]],'float32')
    return noise

def plot_learning_curves(G_loss,D_loss,epochs,label1,label2,name):
    """
    绘制 Generator 和 Discriminator 学习率下降曲线
    """
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,G_loss,n,D_loss)
    #plt.title('loss')
    #plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend([label1,label2],loc='upper right')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def plot_learning_curve(loss,epochs,name):
    """
    绘制 loss 曲线
    """
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def upsampling(im,sx,sy):
    """
    对输入进行上采样,横向采样率为sx,纵向采样率为sy
    """
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def move_to_cpu(t):
    """
    将 tensor 移动到 cpu 上
    """
    t = t.cpu()
    return t

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device=None):
    """
    计算权重衰减率
    """
    alpha = paddle.rand(shape=[1, 1])
    
    alpha = paddle.expand(alpha,shape=real_data.shape)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = paddle.to_tensor(interpolates, dtype='float32', stop_gradient=False)

    disc_interpolates = netD(interpolates)


    gradients = paddle.grad(outputs=disc_interpolates, inputs=interpolates, #怎么改
                                grad_outputs=paddle.ones(disc_interpolates.shape),
                                create_graph=True, retain_graph=True, only_inputs=True)[0] 

    gradient_penalty = ((gradients.norm(2, axis=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def read_image(opt):
    """
    读取图像
    """
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = np2paddle(x,opt)
    x = x[:,0:3,:,:]
    return x

def read_image_dir(dir,opt):
    """
    根据路径dir读取图像
    """
    x = img.imread('%s' % (dir))
    x = np2paddle(x,opt)
    x = x[:,0:3,:,:]
    return x

def np2paddle(x,opt):
    """
    将 numpy.array 转换为 tensor
    """
    if opt.nc_im == 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
    x = paddle.to_tensor(x)
    x = paddle.cast(x, dtype='float32') #float64
    x = norm(x)
    return x

def paddle2uint8(x):
    """
    将 tensor 转换为 numpy.array 
    """
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

def read_image2np(opt):
    """
    读取图像
    """

    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = x[:, :, 0:3]
    return x

def save_networks(netG,netD,z,opt):
    """
    保存 netG 和 netD 的权重, 保存 z 的值
    """
    paddle.save(netG.state_dict(), '%s/netG.pdparams' % (opt.outf))
    paddle.save(netD.state_dict(), '%s/netD.pdparams' % (opt.outf))
    paddle.save(z, '%s/z_opt.pdparams' % (opt.outf))

def adjust_scales2image(real_,opt):
    """
    调整图像尺寸
    """
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3])), 1), opt.scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def adjust_scales2image_SR(real_,opt):
    """
    调整图像尺寸--用于超分辨
    """
    opt.min_size = 18
    opt.num_scales = int((math.log(opt.min_size / min(real_.shape[2], real_.shape[3]), opt.scale_factor_init))) + 1
    scale2stop = int(math.log(min(opt.max_size , max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
    scale2stop = int(math.log(min(opt.max_size, max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def creat_reals_pyramid(real,reals,opt):
    """
    创建图像金字塔
    """
    real = real[:,0:3,:,:]
    for i in range(0,opt.stop_scale+1,1):
        scale = math.pow(opt.scale_factor,opt.stop_scale-i)
        curr_real = imresize(real,scale,opt)
        reals.append(curr_real)
    return reals


def load_trained_pyramid(opt, mode_='train'):
    """
    读取预训练的多尺度模型权重
    """
    mode = opt.mode
    opt.mode = 'train'
    if (mode == 'animation_train') | (mode == 'SR_train') | (mode == 'paint_train'):
        opt.mode = mode
    dir = generate_dir2save(opt)
    if(os.path.exists(dir)):
        Gs = paddle.load('%s/Gs.pdparams' % dir)
        Zs = paddle.load('%s/Zs.pdparams' % dir)
        reals = paddle.load('%s/reals.pdparams' % dir)
        NoiseAmp = paddle.load('%s/NoiseAmp.pdparams' % dir)
    else:
        print('no appropriate trained model is exist, please train first')
    opt.mode = mode
    return Gs,Zs,reals,NoiseAmp

def generate_in2coarsest(reals,scale_v,scale_h,opt):
    """

    """
    real = reals[opt.gen_start_scale]
    real_down = upsampling(real, scale_v * real.shape[2], scale_h * real.shape[3])
    if opt.gen_start_scale == 0:
        in_s = paddle.full(real_down.shape, 0)
    else: #if n!=0
        in_s = upsampling(real_down, real_down.shape[2], real_down.shape[3])
    return in_s

def generate_dir2save(opt):
    """
    生成存储路径
    """
    dir2save = None
    if (opt.mode == 'train') | (opt.mode == 'SR_train'):
        dir2save = 'TrainedModels/%s/scale_factor=%f,alpha=%d' % (opt.input_name[:-4], opt.scale_factor_init,opt.alpha)
    elif (opt.mode == 'animation_train') :
        dir2save = 'TrainedModels/%s/scale_factor=%f_noise_padding' % (opt.input_name[:-4], opt.scale_factor_init)
    elif (opt.mode == 'paint_train') :
        dir2save = 'TrainedModels/%s/scale_factor=%f_paint/start_scale=%d' % (opt.input_name[:-4], opt.scale_factor_init,opt.paint_start_scale)
    elif opt.mode == 'random_samples':
        dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out,opt.input_name[:-4], opt.gen_start_scale)
    elif opt.mode == 'random_samples_arbitrary_sizes':
        dir2save = '%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (opt.out,opt.input_name[:-4], opt.scale_v, opt.scale_h)
    elif opt.mode == 'animation':
        dir2save = '%s/Animation/%s' % (opt.out, opt.input_name[:-4])
    elif opt.mode == 'SR':
        dir2save = '%s/SR/%s' % (opt.out, opt.sr_factor)
    elif opt.mode == 'harmonization':
        dir2save = '%s/Harmonization/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'editing':
        dir2save = '%s/Editing/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'paint2image':
        dir2save = '%s/Paint2image/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
        if opt.quantization_flag:
            dir2save = '%s_quantized' % dir2save
    return dir2save

def post_config(opt):
    """
    修改 config
    """
    # init fixed parameters
    #opt.device = paddle.set_device("cpu" if opt.not_cuda else "gpu:0")
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (opt.input_name[:-4], opt.scale_factor)
    if opt.mode == 'SR':
        opt.alpha = 100

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    paddle.seed(opt.manualSeed)
    if paddle.device.is_compiled_with_cuda() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt

def calc_init_scale(opt):
    """
    计算初始图像的scale
    """
    in_scale = math.pow(1/2,1/3)
    iter_num = round(math.log(1 / opt.sr_factor, in_scale))
    in_scale = pow(opt.sr_factor, 1 / iter_num)
    return in_scale,iter_num

def quant(prev,device):
    """
    对 prev 进行量化(聚类)操作, 返回聚类中心和聚类中心的index
    """
    arr = prev.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, random_state=0).fit(arr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    x = centers[labels]
    x = paddle.to_tensor(x)
    x = paddle.cast(x, dtype='float32')
    x = x.reshape(prev.shape)
    return x,centers

def quant2centers(paint, centers):
    """
    对 paint 进行量化(聚类)操作
    """
    arr = paint.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, init=centers, n_init=1).fit(arr)
    labels = kmeans.labels_
    x = centers[labels]
    x = paddle.to_tensor(x)
    x = paddle.cast(x, dtype='float32') 
    x = x.reshape(paint.shape)
    return x

    return paint


def dilate_mask(mask,opt):
    """
    对 mask 执行形态学膨胀
    """
    if opt.mode == "harmonization":
        element = morphology.disk(radius=7)
    if opt.mode == "editing":
        element = morphology.disk(radius=20)
    mask = paddle2uint8(mask)
    mask = mask[:,:,0]
    mask = morphology.binary_dilation(mask,selem=element)
    mask = filters.gaussian(mask, sigma=5)
    nc_im = opt.nc_im
    opt.nc_im = 1
    mask = np2paddle(mask,opt)
    opt.nc_im = nc_im
    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    plt.imsave('%s/%s_mask_dilated.png' % (opt.ref_dir, opt.ref_name[:-4]), convert_image_np(mask), vmin=0,vmax=1)
    mask = (mask-mask.min())/(mask.max()-mask.min())
    return mask


import paddle.nn as nn
import paddle.nn.functional as F
class ZeroPad2d(nn.Layer):
    """
    2D zero padding类, 对2d数组进行padding操作
    """
    def __init__(self, padding, value=0, data_format="NCHW", name=None):
        """
        构造函数
        """
        super(ZeroPad2d,self).__init__()
        self.padding = padding
        self.data_format = data_format
        self.name = name
        self.value = value

    def forward(self, input):
        """
        前向传播, 对 input 进行2d zero padding操作
        """
        return F.pad(input, self.padding, 'constant', self.value,self.data_format)



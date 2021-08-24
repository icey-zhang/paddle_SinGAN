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

import model.functions as functions
import model.models as models
import os

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.io
import paddle.nn.functional as F

import math
import matplotlib.pyplot as plt
from model.imresize import imresize
from model.functions import ZeroPad2d

def train(opt,Gs,Zs,reals,NoiseAmp):
    """
    训练函数, 训练多尺度模型
    """
    # 读取训练数据
    real_ = functions.read_image(opt)
    in_s = 0
    scale_num = 0
    real = imresize(real_,opt.scale1,opt)
    reals = functions.creat_reals_pyramid(real,reals,opt)
    nfc_prev = 0

    # 依次对各个尺度的模型进行训练
    while scale_num<opt.stop_scale+1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
                pass
        plt.imsave('%s/real_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)
        # 初始化模型
        D_curr,G_curr = init_models(opt)
        if (nfc_prev==opt.nfc):
            # 读取预训练权重
            G_curr.set_state_dict(paddle.load('%s/%d/netG.pdparams' % (opt.out_,scale_num-1)))
            D_curr.set_state_dict(paddle.load('%s/%d/netD.pdparams' % (opt.out_,scale_num-1)))
        # 对单个尺度的模型进行训练
        z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals,Gs,Zs,in_s,NoiseAmp,opt)

        G_curr.eval()
        D_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        # 保存张量
        paddle.save(Zs, '%s/Zs.pdparams' % (opt.out_))
        # paddle.save(Gs, '%s/Gs.pdparams' % (opt.out_))
        paddle.save(reals, '%s/reals.pdparams' % (opt.out_))
        paddle.save(NoiseAmp, '%s/NoiseAmp.pdparams' % (opt.out_))

        scale_num = scale_num + 1 #zjq
        nfc_prev = opt.nfc
        del D_curr,G_curr
    return



def train_single_scale(netD,netG,reals,Gs,Zs,in_s,NoiseAmp,opt,centers=None):
    """
    对单个尺度的模型进行训练
    """
    real = reals[len(Gs)]
    opt.nzx = real.shape[2]
    opt.nzy = real.shape[3]
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    if opt.mode == 'animation_train':
        opt.nzx = real.shape[2]+(opt.ker_size-1)*(opt.num_layer)
        opt.nzy = real.shape[3]+(opt.ker_size-1)*(opt.num_layer)
        pad_noise = 0
    
    m_noise = ZeroPad2d(padding=[int(pad_noise),int(pad_noise),int(pad_noise),int(pad_noise)])
    m_image = ZeroPad2d(padding=[int(pad_image),int(pad_image),int(pad_image),int(pad_image)])

    alpha = opt.alpha

    fixed_noise = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy])
    z_opt = paddle.full(fixed_noise.shape, 0) #怎么改
    z_opt = m_noise(z_opt)

    # 设置优化器
    optimizerD = optim.Adam(parameters=netD.parameters(), learning_rate=opt.lr_d, beta1=opt.beta1, beta2=0.999)
    optimizerG = optim.Adam(parameters=netG.parameters(), learning_rate=opt.lr_g, beta1=opt.beta1, beta2=0.999)
    schedulerD = optim.lr.MultiStepDecay(learning_rate=opt.lr_d,milestones=[1600],gamma=opt.gamma)
    schedulerG = optim.lr.MultiStepDecay(learning_rate=opt.lr_g,milestones=[1600],gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []

    for epoch in range(opt.niter):
        if (Gs == []) & (opt.mode != 'SR_train'):
            z_opt = functions.generate_noise([1,opt.nzx,opt.nzy])
            z_opt = m_noise(paddle.expand(z_opt,shape=[1,3,opt.nzx,opt.nzy]))
            noise_ = functions.generate_noise([1,opt.nzx,opt.nzy])
            noise_ = m_noise(paddle.expand(noise_,shape=[1,3,opt.nzx,opt.nzy]))
        else:
            noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy])
            noise_ = m_noise(noise_)

        ############################
        # (1) 优化 D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            # train with real
            optimizerD.clear_grad()

            output = netD(real)
            # D network 损失函数
            errD_real = -output.mean()#-a
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

            # train with fake
            if (j==0) & (epoch == 0):
                if (Gs == []) & (opt.mode != 'SR_train'):
                    prev = paddle.full([1,opt.nc_z,opt.nzx,opt.nzy], 0)
                    in_s = prev
                    prev = m_image(prev)
                    z_prev = paddle.full([1,opt.nc_z,opt.nzx,opt.nzy], 0)
                    z_prev = m_noise(z_prev)
                    opt.noise_amp = 1
                elif opt.mode == 'SR_train':
                    z_prev = in_s
                    criterion = nn.MSELoss()
                    RMSE = paddle.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    z_prev = m_image(z_prev)
                    prev = z_prev
                else:
                    prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                    prev = m_image(prev)
                    z_prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rec',m_noise,m_image,opt)
                    criterion = nn.MSELoss()
                    RMSE = paddle.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init*RMSE
                    z_prev = m_image(z_prev)
            else:
                prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                prev = m_image(prev)

            if opt.mode == 'paint_train':
                prev = functions.quant2centers(prev,centers)
                plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise = noise_
            else:
                noise = opt.noise_amp*noise_+prev

            fake = netG(noise.detach(),prev)
            output = netD(fake.detach())
            # D network 损失函数
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            
            D_G_z = output.mean().item()

            gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad)
            gradient_penalty.backward(retain_graph=True)

            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()

        errD2plot.append(errD.detach())

        ############################
        # (2) 优化 G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            optimizerG.clear_grad()
            output = netD(fake)
            # G network 损失函数
            errG = -output.mean()
            errG.backward(retain_graph=True)
            if alpha!=0:
                loss = nn.MSELoss()
                if opt.mode == 'paint_train':
                    z_prev = functions.quant2centers(z_prev, centers)
                    plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                Z_opt = opt.noise_amp*z_opt+z_prev
                rec_loss = alpha*loss(netG(Z_opt.detach(),z_prev),real)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt
                rec_loss = 0

            optimizerG.step()

        errG2plot.append(errG.detach()+rec_loss)
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        z_opt2plot.append(rec_loss)

        print('scale %d:[%d/%d],G_loss: %.2f, D_loss: %.2f, rec_loss: %.2f' % (len(Gs), epoch, opt.niter,errG.item(),errD.item(),rec_loss.item()))

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            plt.imsave('%s/fake_sample.png' %  (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt).png'    % (opt.outf),  functions.convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)

            paddle.save(z_opt, '%s/z_opt.pdparams' % (opt.outf))

        schedulerD.step()
        schedulerG.step()
    # 保存模型权重
    functions.save_networks(netG,netD,z_opt,opt)
    return z_opt,in_s,netG    

def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,opt):
    """
    将低一尺度模型的输入与随机噪声进行concatenate
    """
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise])
                    z = paddle.expand(z,shape=[1, 3, z.shape[2], z.shape[3]])
                else:
                    z = functions.generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise])
                z = m_noise(z)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*z+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*Z_opt+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z

def init_models(opt):
    """
    根据 opt 初始化模型
    """
    # generator 初始化:
    netG = models.GeneratorConcatSkip2CleanAdd(opt) #.to(opt.device)
    if opt.netG != '':
        netG.load_state_dict(paddle.load(opt.netG))

    # discriminator 初始化:
    netD = models.WDiscriminator(opt)
    if opt.netD != '':
        netD.load_state_dict(paddle.load(opt.netD))

    return netD, netG

def load_trained_pyramid_New(opt, mode_='train'):
    """
    读取模型预训练权重
    """
    mode = opt.mode
    opt.mode = 'train'
    if (mode == 'animation_train') | (mode == 'SR_train') | (mode == 'paint_train'):
        opt.mode = mode
    dir = functions.generate_dir2save(opt)
    if(os.path.exists(dir)):
        scale_num = 0
        Gs = []
        while scale_num<opt.stop_scale+1:
            # _,G_curr = init_models(opt)
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            G_curr = models.GeneratorConcatSkip2CleanAdd(opt)
            G_curr.eval()
            assert os.path.exists('%s'%dir+'/'+str(scale_num)+'/netG.pdparams')
            G_curr.set_state_dict(paddle.load('%s'%dir+'/'+str(scale_num)+'/netG.pdparams'))
            
            Gs.append(G_curr)
            scale_num = scale_num + 1
        Zs = paddle.load('%s/Zs.pdparams' % dir)
        reals = paddle.load('%s/reals.pdparams' % dir)
        NoiseAmp = paddle.load('%s/NoiseAmp.pdparams' % dir)
    else:
        print('no appropriate trained model is exist, please train first')
    opt.mode = mode
    return Gs,Zs,reals,NoiseAmp

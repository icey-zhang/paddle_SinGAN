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
import paddle.nn as nn
from config import get_arguments
import random
class ConvBlock(nn.Sequential):
    """
    网络基本模块:
    Conv2d-->BatchNorm2D-->LeakyReLU
    """
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_sublayer('conv',nn.Conv2D(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_sublayer('norm',nn.BatchNorm2D(out_channel)),
        self.add_sublayer('LeakyRelu',nn.LeakyReLU(0.2))

def weights_init(m):
    """
    权重初始化
    """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class WDiscriminator(nn.Layer):
    """
    class--判别器类
    """
    def __init__(self, opt):
        """
        构造函数
        """
        super(WDiscriminator, self).__init__()
        #self.is_cuda = paddle.device.is_compiled_with_cuda() #paddle.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_sublayer('block%d'%(i+1),block)
        self.tail = nn.Conv2D(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x):
        """
        前向传播
        """
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Layer):
    """
    class--生成器类
    """
    def __init__(self, opt):
        """
        构造函数
        """
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        #self.is_cuda = paddle.device.is_compiled_with_cuda()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_sublayer('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2D(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )
    def forward(self,x,y):
        """
        前向传播
        """
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x+y

def post_config(opt):
    """
    初始化固定参数
    """
    opt.device = paddle.set_device("cpu" if opt.not_cuda else "cuda:1")
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
    if  paddle.device.is_compiled_with_cuda() and opt.not_cuda: #paddle.cuda.is_available()
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name',default='colusseum.png') #required=True
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt = post_config(opt)
    #model = WDiscriminator(opt)
    model = GeneratorConcatSkip2CleanAdd(opt)
    print(model)
    # conv = ConvBlock(3,32,3,1,1)
    # weights_init(conv)
    # print(conv)
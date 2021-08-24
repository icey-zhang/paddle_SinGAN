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

import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
from config import get_arguments
from model.manipulate import *
from model.training import *
import model.functions as functions
import datetime

if __name__ == '__main__':
    # stratTime = datetime.now()
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name',default='zebra.png') #required=True
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    try:
        os.makedirs(dir2save)
        
    except:
        pass
    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    train(opt, Gs, Zs, reals, NoiseAmp)
    SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt,num_samples=2)

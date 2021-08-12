import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
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

    # if (os.path.exists(dir2save)):
    #     print('trained model already exist')
    # else:
    #     try:
    #         os.makedirs(dir2save)
    #     except OSError:
    #         pass
    #     real = functions.read_image(opt)
    #     functions.adjust_scales2image(real, opt)
    #     train(opt, Gs, Zs, reals, NoiseAmp)
    #     SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)

    try:
        os.makedirs(dir2save)
        
    except:
        pass
    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    train(opt, Gs, Zs, reals, NoiseAmp)
    SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt,num_samples=2)


    ### Debug
    ## save trained pyramid
    # for i in range(len(Gs)):
    #     paddle.save(Gs[i].state_dict(), '%s/Gs'% dir2save+str(i)+'.pdparams' )
    
    # paddle.save(Zs, '%s/Zs.pdparams' % dir2save)
    # paddle.save(reals, '%s/reals.pdparams' % dir2save)
    # paddle.save(NoiseAmp, '%s/NoiseAmp.pdparams' % dir2save)
    # ## save trained pyramid

    # Gs, Zs, reals, NoiseAmp = load_trained_pyramid_New(opt)

    # SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt,num_samples=2)
    ### Debug
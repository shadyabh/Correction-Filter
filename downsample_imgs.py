import utils
import torch
import torch.fft
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import scipy.io as io
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


scale_factor = 4
std = 4.5/np.sqrt(2)
s = utils.get_gauss_flt(32, std).to(device)
s = s/s.sum()

in_dir = '../../SR_testing_datasets/Set14/'
out_dir = './input_x%d/' %scale_factor
out_GT = './GT_x%d/' %scale_factor

imgs = [f for f in os.listdir(in_dir) if '.png' in f]
imgs.sort()

for img in imgs:
    I = utils.load_img_torch(in_dir + img, device)

    if I.shape[2] % scale_factor:
        I = I[:,:,:-(I.shape[2]%scale_factor),:]
    if I.shape[3] % scale_factor:
        I = I[:,:,:,:-(I.shape[3]%scale_factor)]
    utils.save_img_torch(I, out_GT + img)

    y = utils.fft_Down_(I, s, scale_factor)
    
    y_np = np.moveaxis(np.array(torch.abs(y)[0,:].cpu()), 0, 2)
    utils.save_img_torch(torch.abs(y), out_dir + '/PNG/' + img[:-4] + '_Gauss_std%1.1f_x%d_s.png'%(std, scale_factor))
    io.savemat(out_dir + img[:-4] + '_Gauss_std%1.1f_x%d_s.mat' %(std, scale_factor), {'img': y_np})

    I_PIL = transforms.ToPILImage()(I[0,:].cpu())
    W, H = I_PIL.size
    I_PIL_bic_down = I_PIL.resize((W//scale_factor, H//scale_factor), Image.BICUBIC)
    I_PIL_bic_down.save(out_dir + '/bicubic/' + img[:-4] + '_bicubic_down_PIL.png')
    
    S = utils.fft_torch(s, y.shape[2:4])
    s_ = torch.roll(torch.fft.ifftn(S, dim=(-2,-1)).real, (S.shape[2]//2, S.shape[3]//2), dims=(2,3))
    utils.save_img_torch(s_/s_.max(), out_dir + '/Filters/' + img[:-4] + '_Gauss_std%1.1f_x%d_s.png' %(std, scale_factor))

import numpy as np
import torch
import torchvision.transforms as transforms
import utils
import os
import Config
import correction_func
import matplotlib.pyplot as plt
import scipy.io as io

from PIL import Image

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', default='./input_x4/4.5/', type=str)
parser.add_argument('--out_dir', default='./output/corrected_x4/', type=str)
parser.add_argument('--opt_suffix', default='', type=str)
parser.add_argument('--scale_factor', type=int, default=4)
parser.add_argument('--eps', type=float, default=0)
args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################

# Define the reconstruction basis here
r = utils.get_bicubic(args.scale_factor).to(args.device)
r = r/r.sum()

# Define the sampling basis here
sigma = 4.5/np.sqrt(2)
s_size = 32 + args.scale_factor%2
s = utils.get_gauss_flt(s_size, sigma).to(args.device)
s = s/s.sum()

#############################################

if(not os.path.isdir(args.out_dir)):
    os.mkdir(args.out_dir)

imgs = [f for f in os.listdir(args.in_dir) if os.path.isfile(os.path.join(args.in_dir, f)) and ('.mat' in f)]
imgs.sort()

for img_in in imgs:
    y = np.moveaxis(io.loadmat(args.in_dir + img_in)['img'], 2, 0)
    y = torch.tensor(y.real).float().unsqueeze(0).to(args.device)

    Corr_flt = correction_func.Correction_Filter(s, args.scale_factor, (y.shape[2]*args.scale_factor, y.shape[3]*args.scale_factor), r=r, eps=args.eps, inv_type='Tikhonov')
    
    if y.shape[1] == 1:
        y = y.repeat(1,3,1,1)
    img = img_in[0:-4] + '_x%d_corr.png' %(args.scale_factor)

    y_h = Corr_flt.correct_img(y)

    utils.save_img_torch(y_h.real, args.out_dir + img[0:-4] + '_corrected.png', clamp=True)
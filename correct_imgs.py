import numpy as np
import torch
import utils
import os
import Config
import correction_func

conf = Config.Config()

#############################################

out_dir = './' # Set the output directory
in_dir = './' # Set the input directory

# Define the reconstruction basis here
r_np = utils.get_bicubic(conf.bic_size, conf.scale)
r_np = r_np/r_np.sum()
r = torch.tensor(r_np).float().unsqueeze(0).unsqueeze(0).to(conf.device)

# Define the sampling basis here
sigma = 2.5/np.sqrt(2) if conf.scale == 2 else 4.5/np.sqrt(2)  
s_size = np.array([32 - np.mod(conf.scale,2), 32 - np.mod(conf.scale,2)])
s = utils.get_gauss_flt(s_size, sigma)
s_torch = torch.tensor(s).float().unsqueeze(0).unsqueeze(0).to(conf.device)
flt_type = 'Gauss_std%1.1f_x%d' %(sigma, conf.scale)

#############################################

conf.out_dir = out_dir
if(not os.path.isdir(out_dir)):
    os.mkdir(out_dir)

imgs = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f)) and ('.png' in f or '.jpg' in f)]
imgs.sort()

H = lambda I, s, eps: correction_func.correct_img_torch(I, conf.scale, r, s, conf.device, for_dag=True, eps=eps, pad='replicate')

for img_in in imgs:
    y_full = utils.load_img(in_dir + img_in, conf.device)
    if y_full.shape[1] == 1:
        y_full = y_full.repeat(1,3,1,1)
    img = img_in[0:-4] + '_x%d_corr.png' %(conf.scale)

    y_h = H(y_full, s_torch, conf.eps)
    
    utils.save_imag(y_h, out_dir + img[0:-4] + '_corrected.png')

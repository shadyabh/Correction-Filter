import numpy as np
import torch
import utils
import os
import Config
import correction_func

#############################################

# Import the SR network here (e.g. DBPN):
from dbpn_iterative import Net as DBPNITER

# Define the desired SR model here (e.g. DBPN):
SR_model = DBPNITER(num_channels=3, base_filter=64,  feat = 256, num_stages=3, scale_factor=conf.scale)
SR_model = torch.nn.DataParallel(SR_model, device_ids=[conf.gpu], output_device=conf.device)
state_dict = torch.load('./models/DBPN-RES-MR64-3_%dx.pth' %conf.scale, map_location=conf.gpu)
SR_model.load_state_dict(state_dict)
SR_model = SR_model.module
SR_model = SR_model.eval()
R_dag = lambda I: SR_model(I) + torch.nn.functional.interpolate(I, scale_factor=conf.scale, mode='bicubic')

out_dir = './' # Set the output directory
in_dir = './' # Set the input directory

#############################################

conf = Config.Config()
conf.out_dir = out_dir
if(not os.path.isdir(out_dir)):
    os.mkdir(out_dir)


imgs = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f)) and ('.png' in f or '.jpg' in f)]
imgs.sort()

r_np = utils.get_bicubic(conf.bic_size, conf.scale)
r_np = r_np/r_np.sum()
r = torch.tensor(r_np).float().unsqueeze(0).unsqueeze(0).to(conf.device)

R_bic = lambda I: utils.bicubic_up(I, conf.scale, conf.device)
R_bic_conj = lambda I: utils.downsample_bicubic(I, conf.scale, conf.device)
S_conj = lambda I, s, pad: utils.downsample_using_h(I, s, conf.scale, conf.device, pad)
H = lambda I, s, eps: correction_func.correct_img_torch(I, conf.scale, r, s, conf.device, for_dag=True, eps=eps, pad='replicate')
F = lambda I, h, pad: utils.filter_2D_torch(I, h, conf.device, pad=pad)

for img_in in imgs:
    y_full = utils.load_img(in_dir + img_in, conf.device)
    if y_full.shape[1] == 1:
        y_full = y_full.repeat(1,3,1,1)
    img = img_in[0:-4] + '_x%d_corr.png' %(conf.scale)
     
    if(not os.path.isdir(out_dir + '/Filters/')):
            os.mkdir(out_dir + '/Filters/')
    log_file = open(out_dir + '/Filters/' + img[:-4] + '.txt', 'w')
   
    opt_s = correction_func.est_corr(img, y_full, R_dag, S_conj, F, H, conf, log_file)

    y_h = H(y_full, opt_s, conf.eps)
    utils.save_imag(y_h, out_dir + img[0:-4] + '_est.png')
    utils.save_imag(opt_s/opt_s.max(), out_dir + '/Filters/' + img[0:-4] + '_est_s.png')
    log_file.close()

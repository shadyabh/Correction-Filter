import numpy as np
import torch
import torch.fft
import utils
import os
import correction_func
import scipy.io as io
import argparse

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =False

parser = argparse.ArgumentParser()
parser.add_argument('--scale_factor', type=int, default=2)
parser.add_argument('--iterations', type=int, default=500)
parser.add_argument('--lr_s', type=float, default=1e-4)
parser.add_argument('--out_dir', default='./output/estimated_x2/', type=str)
parser.add_argument('--in_dir', default='./input_x2/', type=str)
parser.add_argument('--opt_suffix', default='', type=str)
parser.add_argument('--eps', type=float, default=0)
parser.add_argument('--per_std', type=float, default=0.005)
parser.add_argument('--lambda_l0', type=float, default=1)
parser.add_argument('--crop', type=int, default=150)
parser.add_argument('--save_trace', action='store_true')
parser.add_argument('--suffix', default='', type=str)

args = parser.parse_args()

args.gpu = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device = torch.device(args.gpu)

#############################################

# Import the SR network here (e.g. DBPN):
from dbpn_iterative import Net as DBPNITER

# Define the desired SR model here (e.g. DBPN):
SR_model = DBPNITER(num_channels=3, base_filter=64,  feat = 256, num_stages=3, scale_factor=args.scale_factor)
SR_model = torch.nn.DataParallel(SR_model, device_ids=[args.gpu], output_device=args.device)
state_dict = torch.load('./models/DBPN-RES-MR64-3_%dx.pth' %args.scale_factor, map_location=args.gpu)
SR_model.load_state_dict(state_dict)
SR_model = SR_model.module
SR_model = SR_model.eval()
r = utils.get_bicubic(args.scale_factor).to(args.device)
r = r/r.sum()
R_dag = lambda I: SR_model(I) + args.scale_factor**2 * torch.abs(utils.fft_Up_(I, r, args.scale_factor))

#############################################

if(not os.path.isdir(args.out_dir)):
    os.mkdir(args.out_dir)

imgs = [f for f in os.listdir(args.in_dir) if os.path.isfile(os.path.join(args.in_dir, f)) and ('.mat' in f) or ('.png' in f)]
imgs.sort()

S_conj = lambda I, s: utils.fft_Down_(I, utils.flip_torch(s), args.scale_factor)


ref_bic = None

for img_in in imgs:
    if '.mat' in img_in:
        y_full = io.loadmat(args.in_dir + img_in)['img']
        y_full = torch.tensor(np.moveaxis(y_full, 2, 0)).unsqueeze(0).to(args.device).float()
    else:
        y_full = utils.load_img_torch(args.in_dir + img_in, args.device)
    if y_full.shape[1] == 1:
        y_full = y_full.repeat(1,3,1,1)
    if y_full.shape[1] == 4:
        y_full = y_full[:,:-1,:]
    
    img = img_in[0:-4] + '_x%d_corrected.png' %(args.scale_factor)
     
    if(not os.path.isdir(args.out_dir + '/Filters/')):
            os.mkdir(args.out_dir + '/Filters/')
    
    log_file = open(args.out_dir + '/Filters/' + img[:-4] + args.suffix + '.txt', 'w')

    # Estimate the anti-aliasing filter using the proposed algorithm
    opt_s = correction_func.est_corr(img, y_full, R_dag, S_conj, args, log_file, ref_bic=ref_bic)

    # Apply the correction filter using the estimated anti-aliasing filter
    corr_flt = correction_func.Correction_Filter(opt_s/opt_s.sum(), args.scale_factor, (y_full.shape[2]*args.scale_factor, y_full.shape[3]*args.scale_factor), inv_type='Tikhonov', eps=args.eps)
    y_h = corr_flt.correct_img(y_full)

    # Save the result
    utils.save_img_torch(y_h.real, args.out_dir + img[0:-4] + args.suffix + '_est.png')
    utils.save_img_torch(torch.roll(opt_s.real, ((opt_s.shape[2]//2), (opt_s.shape[3]//2)), dims=(-2,-1))/opt_s.real.max(), args.out_dir + '/Filters/' + img[0:-4] + args.suffix + '_est_s.png')
    log_file.close()

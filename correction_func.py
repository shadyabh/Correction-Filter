import torch
import torch.fft
import numpy as np
import utils

from torch.optim.lr_scheduler import StepLR

class Correction_Filter():
    def __init__(self, s, scale_factor, x_shape, eps=0, r=None, inv_type='naive'):
        self.s = s.clone()
        self.r = None
        if r != None:
            self.r = r.clone()
        else:
            self.r = utils.get_bicubic(scale_factor).float().to(s.device)
            self.r = self.r/self.r.sum()
        self.shape = x_shape
        self.scale_factor = scale_factor
        self.eps = eps
        self.inv_type = inv_type
        self.H = self.find_H(self.s, self.r)
        
    def correct_img(self, y):
        y_h = utils.fft_Filter_(y, self.H)
        return y_h

    def correct_img_(self, y, s):
        self.H = self.find_H(s, self.r)
        y_h = utils.fft_Filter_(y, self.H)
        return y_h  
    
    def find_H(self, s, r):
        R = utils.fft_torch(r, self.shape)
        S = utils.fft_torch(s, self.shape)

        R, S = utils.shift_by(R, 0.5*(not self.scale_factor%2)), utils.shift_by(S, 0.5*(not self.scale_factor%2))

        # Find Q = S*R
        Q = S.conj() * R
        q = torch.fft.ifftn(Q, dim=(-2,-1))
        
        q_d = q[:,:,0::self.scale_factor,0::self.scale_factor]
        Q_d = torch.fft.fftn(q_d, dim=(-2,-1))

        # Find R*R
        RR = R.conj() * R
        rr = torch.fft.ifftn(RR, dim=(-2,-1))
        rr_d = rr[:,:,0::self.scale_factor,0::self.scale_factor]
        RR_d = torch.fft.fftn(rr_d, dim=(-2,-1))

        # Invert S*R
        Q_d_inv = utils.dagger(Q_d, self.eps, mode=self.inv_type)

        H = RR_d * Q_d_inv

        return H

def est_corr(img_name, y_full, R_dag, S_conj, args, log_file=None, ref_bic=None, s_target=None):
    # Crop to area with the most high frequency texture
    crop = min(args.crop, min(y_full.shape[2], y_full.shape[3]))
    topleft_x, topleft_y = utils.crop_high_freq(y_full, crop, args.device)
    print('crop (x,y) = (%d, %d)' %(topleft_x, topleft_y))

    init_sd1 = utils.get_bicubic(1, (31, 31)).to(args.device)
    init_sd1= init_sd1/init_sd1.sum()
    s_d_1 = torch.autograd.Variable(init_sd1, requires_grad=True) 
    
    init_sd2 = utils.get_bicubic(1, (31, 31)).to(args.device)
    init_sd2 = init_sd2/init_sd2.sum()
    s_d_2 = torch.autograd.Variable(init_sd2, requires_grad=True) 
    
    init_sd3 = utils.get_bicubic(1, (31, 31)).to(args.device)
    init_sd3 = init_sd3/init_sd3.sum()
    s_d_3 = torch.autograd.Variable(init_sd3, requires_grad=True) 

    init_sd4 = utils.get_bicubic(args.scale_factor, (32, 32)).to(args.device)
    init_sd4 = init_sd4/init_sd4.sum()
    s_d_4 = torch.autograd.Variable(init_sd4, requires_grad=True) 
    optimizer_sd = torch.optim.Adam([{'params' : s_d_1, 'lr': args.lr_s}, {'params' : s_d_2, 'lr': args.lr_s}, {'params' : s_d_3, 'lr': args.lr_s}, {'params' : s_d_4, 'lr': args.lr_s}])

    objective = torch.nn.L1Loss()

    s_c = torch.fft.ifftn(utils.fft_torch(s_d_1/s_d_1.sum(), y_full.shape[2:4])*utils.fft_torch(s_d_2/s_d_2.sum(), y_full.shape[2:4])*
                          utils.fft_torch(s_d_3/s_d_3.sum(), y_full.shape[2:4])*utils.fft_torch(s_d_4/s_d_4.sum(), y_full.shape[2:4]) ,dim=(-2,-1))
    with torch.no_grad():
        corr_flt = Correction_Filter(s_c, args.scale_factor, (y_full.shape[2]*args.scale_factor, y_full.shape[3]*args.scale_factor), inv_type='Tikhonov', eps=0)

    for itr in range(args.iterations):
        optimizer_sd.zero_grad()

        s_c = torch.fft.ifftn(utils.fft_torch(s_d_1/s_d_1.sum(), y_full.shape[2:4])*utils.fft_torch(s_d_2/s_d_2.sum(), y_full.shape[2:4])*
                          utils.fft_torch(s_d_3/s_d_3.sum(), y_full.shape[2:4])*utils.fft_torch(s_d_4/s_d_4.sum(), y_full.shape[2:4]) ,dim=(-2,-1))

        y_full_h = torch.abs(corr_flt.correct_img_(y_full, s_c)).float()
        y_h = y_full_h[:,:,topleft_y:topleft_y+crop, topleft_x:topleft_x+crop]

        x_hat = R_dag(y_h + torch.randn_like(y_h)*args.per_std) 

        x_hat1 = utils.fft_Filter_(x_hat,  utils.fft_torch(utils.flip_torch(s_d_1)/s_d_1.sum(), s = x_hat.shape[2:4]))
        x_hat2 = utils.fft_Filter_(x_hat1, utils.fft_torch(utils.flip_torch(s_d_2)/s_d_2.sum(), s = x_hat1.shape[2:4]))
        x_hat3 = utils.fft_Filter_(x_hat2, utils.fft_torch(utils.flip_torch(s_d_3)/s_d_3.sum(), s = x_hat2.shape[2:4]))

        y_hat = torch.abs(S_conj(x_hat3, s_d_4/s_d_4.sum()))
        
        shave = ((crop - y_hat.shape[2])//2, (crop - y_hat.shape[3])//2 )
        y = y_full[:,:,topleft_y+shave[0]:topleft_y+crop-shave[0], topleft_x+shave[1]:topleft_x+crop-shave[1]]
        consistency = objective(y_hat[:,:,2:-2, 2:-2], y[:,:,2:-2, 2:-2])
        with torch.no_grad():
            x_c, y_c = utils.get_center_of_mass(torch.roll(s_c.real, (s_c.shape[2]//2, s_c.shape[3]//2), dims=(-2,-1)), args.device)
        
        abs_s = torch.abs(s_c)
        l0 = torch.mean( abs_s[abs_s > 0]**0.5 ) # Relaxed l_0
        loss = consistency + args.lambda_l0*l0
        
        loss.backward()
        optimizer_sd.step()

        with torch.no_grad():
            if(args.save_trace and np.mod(itr, 10) == 0):
                utils.save_img_torch(y_full_h, args.out_dir + 'within_loop.png')           

            opt_s = s_c.clone()
            s_norm = s_c/s_c.sum()
            out_log = img_name[:-4] + \
                '| Itr = %d' %itr + \
                '| loss = %.7f' %(loss.item()) + \
                '| x_c, y_c = %.2f/%.2f, %.2f/%.2f' %(x_c, (s_c.shape[3] - 1)/2, y_c, (s_c.shape[2] - 1)/2)
            if not ref_bic == None:
                out_log += '| PSNR bic = %.5f' %(-10*torch.log10( torch.mean( (y_full_h - ref_bic)**2 ) ))
            if not s_target == None:
                l_s_test = torch.sum( torch.abs(s_norm - s_target)).item()
                out_log += '| SAE(s) = %.3f' %l_s_test
            if not log_file == None:
                log_file.write(out_log + '\n')
            print(out_log)
    return opt_s

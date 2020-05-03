import torch
import numpy as np
import utils

def correct_img_torch(x_s, scale, r, s, device, for_dag = True, eps = 1e-9, pad='circular'):
    conv_shape = (s.shape[2] + r.shape[2] - 1, s.shape[3] + r.shape[3] - 1)
    S = utils.fft2(s/s.sum(), conv_shape[1], conv_shape[0])
    R = utils.fft2(utils.flip(r)/r.sum(), conv_shape[1], conv_shape[0])
    Q_unscaled = utils.mul_complex(R, S)
    q_unscaled = torch.irfft(Q_unscaled, signal_ndim=2, normalized=False, onesided=False)
    q = q_unscaled[:,:,np.mod(q_unscaled.shape[2], scale)::scale, np.mod(q_unscaled.shape[3], scale)::scale]
    Q = torch.rfft(q, signal_ndim=2, normalized=False, onesided=False)

    # Q_star = utils.conj(Q)
    # abs2_Q = utils.abs2(Q)
    # H = torch.cat( (Q_star[:,:,:,:,0:1]/(abs2_Q[:,:,:,:,0:1] + eps), Q_star[:,:,:,:,1:2]/(abs2_Q[:,:,:,:,0:1] + eps)), dim=4)

    H = utils.inv_complex(Q, eps)

    h_ = torch.irfft(H, signal_ndim=2, normalized=False, onesided=False)
    h = utils.roll_y(utils.roll_x(h_/h_.sum(), -1), -1)

    x_h = utils.filter_2D_torch(x_s, utils.flip(h), device, pad=pad)

    if(for_dag):
        x_h = utils.bicubic_up(x_h, scale, device)
        x_h = utils.downsample_bicubic(x_h, scale, device)

    return x_h

def est_corr(img_name, y_full, R_dag, S_conj, F, H, conf, log_file=None, ref_bic=None, s_target=None):
    crop = min(conf.crop, min(y_full.shape[2], y_full.shape[3]))
    topleft_x, topleft_y = utils.crop_high_freq(y_full, crop, conf.device)
    print('crop (x,y) = (%d, %d)' %(topleft_x, topleft_y))

    init_s1 = torch.tensor(utils.get_bicubic(conf.base_s_size + 1, conf.scale)).float().unsqueeze(0).unsqueeze(0).to(conf.device)
    s1 = torch.autograd.Variable(init_s1, requires_grad=True) 
    optimizer_s1 = torch.optim.Adam([s1], conf.lr_s, amsgrad=False)

    init_s2 = torch.tensor(utils.get_bicubic(conf.base_s_size + 1, conf.scale)).float().unsqueeze(0).unsqueeze(0).to(conf.device)
    s2 = torch.autograd.Variable(init_s2, requires_grad=True) 
    optimizer_s2 = torch.optim.Adam([s2], conf.lr_s, amsgrad=False)

    init_s3 = torch.tensor(utils.get_bicubic(conf.base_s_size + 1, conf.scale)).float().unsqueeze(0).unsqueeze(0).to(conf.device)
    s3 = torch.autograd.Variable(init_s3, requires_grad=True) 
    optimizer_s3 = torch.optim.Adam([s3], conf.lr_s, amsgrad=False)

    init_s4 = torch.tensor(utils.get_bicubic(conf.base_s_size, conf.scale)).float().unsqueeze(0).unsqueeze(0).to(conf.device)
    s4 = torch.autograd.Variable(init_s4, requires_grad=True) 
    optimizer_s4 = torch.optim.Adam([s4], conf.lr_s, amsgrad=False)

    for itr in range(conf.iterations):
        optimizer_s1.zero_grad()
        optimizer_s2.zero_grad()
        optimizer_s3.zero_grad()
        optimizer_s4.zero_grad()

        s = utils.equiv_flt_conv(s4, 
            utils.equiv_flt_conv(s3, 
            utils.equiv_flt_conv(s2, s1)
            ) # s3
            ) # s4
        
        y_full_h = H(y_full, s, conf.eps)
        y_h = y_full_h[:,:,topleft_y:topleft_y+crop, topleft_x:topleft_x+crop]
        
        if(conf.out_curr_corr and np.mod(itr, 10) == 0):
            utils.save_imag(y_full_h, conf.out_dir + 'in_loop.png')

        if(conf.gpu == "cpu"):
            with torch.no_grad():
                x_hat = R_dag(y_h) 
        else:
            x_hat = R_dag(y_h) 
        
        y_hat = S_conj(x_hat, s, False)

        shave = ((crop - y_hat.shape[2])//2, (crop - y_hat.shape[3])//2 )
        y = y_full[:,:,topleft_y+shave[0]:topleft_y+crop-shave[0], topleft_x+shave[1]:topleft_x+crop-shave[1]]
        consistency = conf.l1(y_hat, y)

        with torch.no_grad():
            x_c, y_c = utils.get_center_of_mass(s, conf.device)
        l_bound = conf.bound_loss(s)
        l0 = conf.l1(torch.abs(s[s!=0])**conf.l0_pow, torch.zeros_like(s[s!=0]))
        l_center = torch.sum( (x_c - (4*conf.base_s_size-1)/2)**2 + (y_c - (4*conf.base_s_size-1)/2)**2 )
        loss = consistency + conf.lambda_bound*l_bound + conf.lambda_l0*l0 + conf.lambda_center*l_center
        loss.backward()
        optimizer_s1.step()
        optimizer_s2.step()
        optimizer_s3.step()
        optimizer_s4.step()

        with torch.no_grad():
            opt_s = s.clone()
            s_norm = s/s.sum()
            out_log = img_name[:-4] + \
                '| Itr = %d' %itr + \
                '| loss = %.7f' %(loss.item()) + \
                '| obj = %.7f' %(consistency.item()) + \
                '| x_c, y_c, 0 = %.2f, %.2f, %.2f' %(x_c, y_c, (4*conf.base_s_size-1)/2)
            if not ref_bic == None:
                out_log += '| PSNR bic = %.5f' %(-10*torch.log10( torch.mean( (y_full_h - ref_bic)**2 ) ))
            if not s_target == None:
                l_s_test = torch.sum( torch.abs(s_norm - s_target)).item()
                out_log += '| SAE(s) = %.3f' %l_s_test
            if not log_file == None:
                log_file.write(out_log + '\n')
            print(out_log)
    return opt_s

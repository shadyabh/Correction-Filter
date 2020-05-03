import os
import torch
import utils

class Config:
    def __init__(self):

        self.lr_s = 1e-3
        self.scale = 4
        self.base_s_size = 16 if self.scale == 2 else 16
        self.bic_size = 64 if self.scale == 2 else 128
        self.eps = 0
        self.iterations = 1000 if self.scale == 2 else 1000
        self.lambda_l0 = 0 if self.scale == 2 else 0.0001
        self.lambda_bound = 1
        self.lambda_center = 0
        self.out_dir = None ## define in estimate_correction.py ##
        self.ref_dir = None ## define in estimate_correction.py ##
        self.gpu = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.gpu)
        self.crop = 140 if self.scale == 2 else 80

        self.l0_pow = 0.2
        self.Huber = torch.nn.SmoothL1Loss()
        self.l1 = torch.nn.L1Loss()
        self.l2 = torch.nn.MSELoss()
        m_s = utils.get_boundaries_mask((4*self.base_s_size, 4*self.base_s_size), (4*self.scale), self.device)
        self.bound_loss = lambda k: self.l1(torch.abs(k*m_s), torch.zeros_like(k))

        self.out_curr_corr = False # Output within loop corrected image
        
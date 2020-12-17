import torch
import torch.fft
import torchvision.transforms as transforms
import numpy as np
from scipy import interpolate
from scipy import fftpack
from scipy import integrate
from scipy import signal
from PIL import Image

def flip(x):
    return x.flip([2,3])

def flip_torch(x):
    x_ = torch.flip(torch.roll(x, ((x.shape[2]//2), (x.shape[3]//2)), dims=(2,3)), dims=(2,3))
    return torch.roll(x_, (- (x_.shape[2]//2), -(x_.shape[3]//2)), dims=(2,3))

def flip_np(x):
    x_ = np.flip(np.roll(x, ((x.shape[0]//2), (x.shape[1]//2)), (0,1)))
    return np.roll(x_, (- (x_.shape[0]//2), -(x_.shape[1]//2)), (0,1))

def shift_by(H, shift):
    k_x = np.linspace(0, H.shape[3]-1, H.shape[3])
    k_y = np.linspace(0, H.shape[2]-1, H.shape[2])

    k_x[((k_x.shape[0] + 1)//2):] -= H.shape[3]
    k_y[((k_y.shape[0] + 1)//2):] -= H.shape[2]

    exp_x, exp_y = np.meshgrid(np.exp(-1j * 2* np.pi * k_x * shift / H.shape[3]), np.exp(-1j * 2* np.pi * k_y * shift / H.shape[2]))

    exp_x_torch = (torch.tensor(np.real(exp_x)) + 1j*torch.tensor(np.imag(exp_x))).unsqueeze(0).unsqueeze(0).to(H.device)
    exp_y_torch = (torch.tensor(np.real(exp_y)) + 1j*torch.tensor(np.imag(exp_y))).unsqueeze(0).unsqueeze(0).to(H.device)

    return H * exp_x_torch * exp_y_torch

def fft_torch(x, s=None, zero_centered=True):
    # s = (Ny, Nx)
    __,__,H,W = x.shape
    if s == None:
        s = (H, W)
    if zero_centered:
        x_ = torch.roll(x, ((H//2), (W//2)), dims=(2,3))
    else:
        x_ = x
    x_pad = torch.nn.functional.pad(x_, (0, s[1] - W, 0, s[0] - H))
    if zero_centered:
        x_pad_ = torch.roll(x_pad, (- (H//2), -(W//2)), dims=(2,3))
    else:
        x_pad_ = x_pad
    return torch.fft.fftn(x_pad_, dim=(-2,-1))

def bicubic_ker(x, y, a=-0.5):
    # X:
    abs_phase = np.abs(x)
    abs_phase2 = abs_phase**2
    abs_phase3 = abs_phase**3
    out_x = np.zeros_like(x)
    out_x[abs_phase <= 1] = (a+2)*abs_phase3[abs_phase <= 1] - (a+3)*abs_phase2[abs_phase <= 1] + 1
    out_x[(abs_phase > 1) & (abs_phase < 2)] = a*abs_phase3[(abs_phase > 1) & (abs_phase < 2)] -\
                                              5*a*abs_phase2[(abs_phase > 1) & (abs_phase < 2)] +\
                                              8*a*abs_phase[(abs_phase > 1) & (abs_phase < 2)] - 4*a
    # Y:
    abs_phase = np.abs(y)
    abs_phase2 = abs_phase**2
    abs_phase3 = abs_phase**3
    out_y = np.zeros_like(y)
    out_y[abs_phase <= 1] = (a+2)*abs_phase3[abs_phase <= 1] - (a+3)*abs_phase2[abs_phase <= 1] + 1
    out_y[(abs_phase > 1) & (abs_phase < 2)] = a*abs_phase3[(abs_phase > 1) & (abs_phase < 2)] -\
                                              5*a*abs_phase2[(abs_phase > 1) & (abs_phase < 2)] +\
                                              8*a*abs_phase[(abs_phase > 1) & (abs_phase < 2)] - 4*a 

    return out_x*out_y

def build_flt(f, size):
    is_even_x = not size[1] % 2 
    is_even_y = not size[0] % 2 

    grid_x = np.linspace(-(size[1]//2 - is_even_x*0.5), (size[1]//2 - is_even_x*0.5), size[1])
    grid_y = np.linspace(-(size[0]//2 - is_even_y*0.5), (size[0]//2 - is_even_y*0.5), size[0])

    x, y = np.meshgrid(grid_x, grid_y)

    h =f(x, y)
    h = np.roll(h, (- (h.shape[0]//2), -(h.shape[1]//2)), (0,1))

    return torch.tensor(h).float().unsqueeze(0).unsqueeze(0)

def get_bicubic(scale, size=None):
    f = lambda x,y: bicubic_ker(x/scale, y/scale)
    if size:
        h = build_flt(f, (size[0], size[1]))
    else:
        h = build_flt(f, (4*scale + 8 + scale%2, 4*scale + 8 + scale%2))
    return h

def get_box(supp, size=None):
    if size == None:
        size = (supp[0]*2, supp[1]*2)

    h = np.zeros(size)

    h[0:supp[0]//2  , 0:supp[1]//2] = 1
    h[0:supp[0]//2  , -(supp[1]//2):] = 1
    h[-(supp[0]//2):, 0:supp[1]//2] = 1
    h[-(supp[0]//2):, -(supp[1]//2):] = 1

    return torch.tensor(h).float().unsqueeze(0).unsqueeze(0)

def get_delta(size):
    h = torch.zeros(1,1,size,size)
    h[0,0,0,0] = 1
    return h

def get_gauss_flt(flt_size, std):
    f = lambda x,y: np.exp( -(x**2 + y**2)/2/std**2 )
    h = build_flt(f, (flt_size,flt_size))
    return h

def fft_Filter_(x, A):
    X_fft = torch.fft.fftn(x, dim=(-2,-1))
    HX = A * X_fft
    return torch.fft.ifftn(HX, dim=(-2,-1))

def fft_Down_(x, h, alpha):
    X_fft = torch.fft.fftn(x, dim=(-2,-1))
    H = fft_torch(h, s=X_fft.shape[2:4])
    HX = H * X_fft
    margin = (alpha - 1)//2
    y = torch.fft.ifftn(HX, dim=(-2,-1))[:,:,margin:HX.shape[2]-margin:alpha, margin:HX.shape[3]-margin:alpha]
    return y

def fft_Up_(y, h, alpha):
    x = torch.zeros(y.shape[0], y.shape[1], y.shape[2]*alpha, y.shape[3]*alpha).to(y.device) 
    H = fft_torch(h, s=x.shape[2:4])
    start = alpha//2
    x[:,:,start::alpha, start::alpha] = y
    X = torch.fft.fftn(x, dim=(-2,-1))
    HX = H * X
    return torch.fft.ifftn(HX, dim=(-2,-1))

def zero_SV(H, eps):
    H_real = H.real
    H_imag = H.imag
    abs_H2 = H_real**2 + H_imag**2 
    H[abs_H2/abs_H2.max() <= eps**2] = 0
    return H

def dagger(X, eps=0, mode='Tikhonov'):    
    real = X.real
    imag = X.imag
    abs2 = real**2 + imag**2
    if mode == 'naive':
        out = X.clone()
        out[abs2/abs2.max() > eps**2] = 1/X[abs2/abs2.max() > eps**2]
        out[abs2/abs2.max() <= eps**2] = 0
        return out 
    if mode == 'Tikhonov':
        return X.conj()/(abs2 + eps**2)

def load_img_torch(dir, device):
    I = Image.open(dir)
    I = transforms.ToTensor()(I).unsqueeze(0)
    return I.to(device)

def save_img_torch(I, dir, clamp=True):
    if clamp:
        img = torch.clamp(I, 0, 1)[0,:].detach().cpu()
    else:
        img = I[0,:].detach().cpu()
    img = transforms.ToPILImage()(img)
    img.save(dir)

def get_center_of_mass(s, device):
    idx_x = torch.linspace(0, s.shape[3]-1, s.shape[3]).to(device)
    idx_y = torch.linspace(0, s.shape[2]-1, s.shape[2]).to(device)
    i_y, i_x = torch.meshgrid(idx_y, idx_x)

    x_c = torch.sum(s*i_x/s.sum())
    y_c = torch.sum(s*i_y/s.sum())

    return x_c, y_c

def crop_high_freq(I, crop_size, device):
    if(crop_size >= I.shape[2] or crop_size >= I.shape[3]):
        argmax_x = I.shape[3]//2 - crop_size//2
        argmax_y = I.shape[2]//2 - crop_size//2
    else:
        filt = torch.tensor([[ 0,-1, 0],
                             [-1, 4,-1],
                             [ 0,-1, 0]]).float().to(device)
        if(I.shape[1] > 1):
            I_gray = 0.2126*I[:,0:1,:,:] + 0.7152*I[:,1:2,:,:] + 0.0722*I[:,2:3,:,:]
        else:
            I_gray = I
        D = torch.abs(torch.nn.functional.conv2d(I_gray, filt.unsqueeze(0).unsqueeze(0)))
        Avg = torch.nn.functional.avg_pool2d(D, crop_size, stride=1, padding=0, ceil_mode=False)
        argmax = Avg.argmax()
        argmax_y = argmax//Avg.shape[3]
        argmax_x = argmax % Avg.shape[3]
    return argmax_x, argmax_y    
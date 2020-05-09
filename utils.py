import torch
import torchvision
import numpy as np
from scipy import interpolate
from scipy import fftpack
from scipy import integrate
from scipy import signal
from PIL import Image
import cv2 as cv

def flip(x):
    return x.flip([2,3])

def roll_x(x, n):
    return torch.cat((x[:,:,:,-n:], x[:,:,:,:-n]), dim = 3)

def roll_y(x, n):
    return torch.cat((x[:,:,-n:,:], x[:,:,:-n,:]), dim = 2)

def bicubic_kernel_2D(x, y, a=-0.5):
    # get X
    abs_phase = np.abs(x)
    abs_phase3 = abs_phase**3
    abs_phase2 = abs_phase**2
    if abs_phase < 1:
        out_x = (a+2)*abs_phase3 - (a+3)*abs_phase2 + 1
    else:
        if abs_phase >= 1 and abs_phase < 2:
            out_x = a*abs_phase3 - 5*a*abs_phase2 + 8*a*abs_phase - 4*a 
        else:
            out_x = 0
    # get Y
    abs_phase = np.abs(y)
    abs_phase3 = abs_phase**3
    abs_phase2 = abs_phase**2
    if abs_phase < 1:
        out_y = (a+2)*abs_phase3 - (a+3)*abs_phase2 + 1
    else:
        if abs_phase >= 1 and abs_phase < 2:
            out_y = a*abs_phase3 - 5*a*abs_phase2 + 8*a*abs_phase - 4*a 
        else:
            out_y = 0

    return out_x*out_y

def get_bicubic(size, scale):
    is_even = not np.mod(size, 2)
    grid_r = np.linspace(-(size//2) + 0.5*is_even,  size//2 - 0.5*is_even, size)
    r = np.zeros((size, size))
    for m in range(size):
        for n in range(size):
            r[m, n] = bicubic_kernel_2D(grid_r[n]/scale, grid_r[m]/scale)
    r = r/r.sum()

    return r

def get_gauss_flt(flt_size, std):
    is_even = 1 - np.mod(flt_size[0], 2)
    grid = np.linspace(- (flt_size[0]//2) + 0.5*is_even, flt_size[0]//2 - 0.5*is_even, flt_size[0])
    h = np.zeros(flt_size)
    for m in range(flt_size[0]):
        for n in range(flt_size[1]):
            h[m, n] = np.exp(-(grid[n]**2 + grid[m]**2)/(2*std**2))
    return h

def downsample_bicubic_2D(I, scale, device):
    # scale: integer > 1
    filter_supp = 4*scale + 2 + np.mod(scale, 2)
    is_even = 1 - np.mod(scale, 2)
    Filter = torch.zeros(1,1,filter_supp,filter_supp).float().to(device)
    grid = np.linspace(-(filter_supp//2) + 0.5*is_even, filter_supp//2 - 0.5*is_even, filter_supp)
    for n in range(filter_supp):
        for m in range(filter_supp):
            Filter[0, 0, m, n] = bicubic_kernel_2D(grid[n]/scale, grid[m]/scale)

    h = Filter/torch.sum(Filter)
    pad = np.int((filter_supp - scale)/2)
    I_padded = torch.nn.functional.pad(I, [pad, pad, pad, pad], mode='circular')
    I_out = torch.nn.functional.conv2d(I_padded, h, stride=(scale, scale))

    return I_out

def downsample_bicubic(I, scale, device):
    out = torch.zeros(I.shape[0], I.shape[1], I.shape[2]//scale, I.shape[3]//scale).to(device)
    out[:,0:1, :, :] = downsample_bicubic_2D(I[:, 0:1, :, :], scale, device)
    if(I.shape[1] > 1):
        out[:,1:2, :, :] = downsample_bicubic_2D(I[:, 1:2, :, :], scale, device)
        out[:,2:3, :, :] = downsample_bicubic_2D(I[:, 2:3, :, :], scale, device)
    return out

def bicubic_up(img, scale, device):
    flt_size = 4*scale + np.mod(scale, 2)
    is_even = 1 - np.mod(scale, 2)
    grid = np.linspace(-(flt_size//2) + 0.5*is_even, flt_size//2 - 0.5*is_even, flt_size)
    Filter = torch.zeros(1,1,flt_size, flt_size).to(device)
    for m in range(flt_size):
        for n in range(flt_size):
            Filter[0, 0, m, n] =  bicubic_kernel_2D(grid[n]/scale, grid[m]/scale)
    h = flip(Filter)
    pad = 1
    x_pad = torch.nn.functional.pad(img, [pad, pad, pad, pad], mode='circular')
    img_up_torch = torch.nn.functional.interpolate(img, scale_factor=scale, mode='bicubic')
    img_up = torch.zeros_like(img_up_torch)
    for ch in range(img.shape[1]):
        img_up[:,ch:ch+1,:,:] = torch.nn.functional.conv_transpose2d(x_pad[:,ch:ch+1,:,:], h, stride=scale,
            padding=(flt_size//2 + np.int(np.ceil(scale/2)), flt_size//2 + np.int(np.ceil(scale/2))))

    return img_up#, img_up_torch

def filter_2D_torch(I, Filter, device, pad = False):
    h = (Filter)/Filter.sum()
    if pad:
        pad_y = Filter.shape[2] // 2
        pad_x = Filter.shape[3] // 2
        I_pad = torch.nn.functional.pad(I, (pad_x, pad_x, pad_y, pad_y), mode=pad)
    else:
        I_pad = I
    batch, C, H, W = I_pad.shape
    out = torch.nn.functional.conv2d(I_pad.view(-1, 1, H, W), h)

    return out.view(batch, C, out.shape[2], out.shape[3])

def downsample_using_h(I_in, Filter, scale, device, pad=False):
    filter_supp_x = Filter.shape[3]
    filter_supp_y = Filter.shape[2]
    h = (Filter)/torch.sum(Filter)
    if pad:
        pad_x = np.int((filter_supp_x - scale)/2)
        pad_y = np.int((filter_supp_y - scale)/2)
        I_padded = torch.nn.functional.pad(I_in, [pad_x, pad_x, pad_y, pad_y], mode=pad)
    else:
        pad_x = scale//2
        pad_y = scale//2
        I_padded = torch.nn.functional.pad(I_in, [pad_x, pad_x, pad_y, pad_y], mode='constant')
    batch, c, H, W = I_padded.shape
    I_out = torch.nn.functional.conv2d(I_padded.view(-1,1,H,W), h, stride=scale)
    return I_out.view(batch, c, I_out.shape[2], I_out.shape[3])
    
def fft2(s, n_x = [], n_y = []):
    h, w = s.shape[2], s.shape[3]
    if(n_x == []):
        n_x = w
    if(n_y == []):
        n_y = h 
    s_pad = torch.nn.functional.pad(s, (0, n_x - w, 0 , n_y - h))

    return torch.rfft(s_pad, signal_ndim=2, normalized=False, onesided=False)

def mul_complex(t1, t2):
    ## Re{Z0 * Z1} = a0*a1 - b0*b1
    out_real = t1[:,:,:,:,0:1]*t2[:,:,:,:,0:1] - t1[:,:,:,:,1:2]*t2[:,:,:,:,1:2]
    ## Im{Z0 * Z1} = i*(a0*b1 + b0*a1)
    out_imag = t1[:,:,:,:,0:1]*t2[:,:,:,:,1:2] + t1[:,:,:,:,1:2]*t2[:,:,:,:,0:1]
    return torch.cat((out_real, out_imag), dim=4)

def conj(x):
    out = x.clone()
    out[:,:,:,:,1] = -out[:,:,:,:,1]
    return out 

def abs2(x):
    out = torch.zeros_like(x)
    out[:,:,:,:,0] = x[:,:,:,:,0]**2 + x[:,:,:,:,1]**2
    return out

def inv_complex(x, eps=0):
    real = x[:,:,:,:,0]
    imag = x[:,:,:,:,1]
    abs2 = real**2 + imag**2
    out_real = real.clone()
    out_imag = imag.clone()
    out_real[abs2 > eps] = real[abs2 > eps] / abs2[abs2 > eps]
    out_imag[abs2 > eps] = - imag[abs2 > eps] / abs2[abs2 > eps]
    out_real[abs2 <= eps] = 0
    out_imag[abs2 <= eps] = 0

    return torch.cat((out_real.unsqueeze(4), out_imag.unsqueeze(4)), dim = 4)

def tensorImg2npImg(I):
    return np.moveaxis(np.array(I[0,:].detach().cpu()), 0, 2)

def save_imag(I, dir):
    I = torch.clamp(I, 0, 1)
    I_np = tensorImg2npImg(I)
    if(I_np.shape[2] == 1):
        I_PIL = Image.fromarray(np.uint8(I_np[:,:,0]*255))
    else:
        I_PIL = Image.fromarray(np.uint8(I_np*255))
    I_PIL.save(dir)

def load_img(dir, device):
    I_PIL = Image.open(dir)
    I_np = np.array(I_PIL)/255.0
    if(np.shape(I_np.shape)[0] < 3):
        I = torch.tensor(I_np).float().unsqueeze(0).unsqueeze(0).to(device)
    else:
        I_np = I_np[:,:,0:3]
        I = torch.tensor(np.moveaxis(I_np, 2, 0)).float().unsqueeze(0).to(device)
    return I

def get_boundaries_mask(size, sigma, device):
    is_even = not np.mod(size[0], 2)
    grid_y = np.linspace(-(size[0]//2) + 0.5*is_even,  size[0]//2 - 0.5*is_even, size[0])
    is_even = not np.mod(size[1], 2)
    grid_x = np.linspace(-(size[1]//2) + 0.5*is_even,  size[1]//2 - 0.5*is_even, size[1])
    mask = np.zeros(size)
    for m in range(size[0]):
        for n in range(size[1]):
            mask[m, n] = 1 - np.exp( -(grid_x[n]**2 + grid_y[m]**2)/(2*sigma**2) )
    return torch.tensor(mask).float().unsqueeze(0).unsqueeze(0).to(device)   

def get_center_of_mass(s, device):
    idx_x = torch.linspace(0, s.shape[3]-1, s.shape[3]).to(device)
    idx_y = torch.linspace(0, s.shape[2]-1, s.shape[2]).to(device)
    i_x, i_y = torch.meshgrid(idx_x, idx_y)

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

def equiv_flt_conv(s1, s2):
    return torch.nn.functional.conv2d(s1, flip(s2), padding=(s2.shape[2] - 1, s2.shape[3]-1))



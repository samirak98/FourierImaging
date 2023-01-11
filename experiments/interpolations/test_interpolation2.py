import torch
from torchvision import transforms
import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("WebAgg")

#%% custom imports
import sys, os
sys.path.append(os.path.abspath('../../'))

import fourierimaging as fi

from fourierimaging.utils.helpers import load_model, init_opt, fix_seed
from fourierimaging.modules import TrigonometricResize_2d
from fourierimaging.utils import datasets as data
import fourierimaging.train as train

plt.close('all')

def nice_imshow(img):
    plt.figure()
    plt.imshow(img)
    plt.colorbar()

#%%
N = 3
img = torch.rand([2*N + 1, 2*N + 1])
coeffs = torch.fft.fftshift(torch.fft.fft2(img, norm = 'forward'))
coeffs.imag[[0,-1],:] = 0 #= coeffs[[0,-1],:].real
#coeffs.imag[:,[0,-1]] = 0 # = coeffs[:,[0,-1]].real
#img = torch.fft.ifft2(torch.fft.ifftshift(coeffs), norm = 'forward')

#nice_imshow(img)

def trigo_val(x, coeffs):
    N = coeffs.shape[0]//2
    M = coeffs.shape[1]//2
    res = 0
    for k in range(2*N + 1):
        for l in range(2*M+1):
            res += coeffs[k,l]*torch.exp(1j * 2 * np.pi * ((k-N) * x[...,0] + (l-M) * x[...,1]))
    return res

s_x = np.linspace(0,1, 2*N+1, endpoint=False)
s_y = np.linspace(0,1, 2*N, endpoint=False)
X,Y = np.meshgrid(s_x,s_y)
grid = np.stack([X,Y]).T
grid = torch.tensor(grid)

v_eval = trigo_val(grid, coeffs)
v_eval_fft = torch.fft.fftshift(torch.fft.fft2(v_eval, norm = 'forward'))

img_fft = torch.fft.fftshift(torch.fft.fft2(img, norm = 'forward'))


#%%
c = torch.zeros((2*N+1,), dtype=torch.cfloat)
c[:] = 0.5*v_eval_fft[:,0]

#c[-1] = c[0]
#c = torch.flip(c,dims=[-1])
#c = torch.flip(torch.conj(c),dims=[-1])
#c = (c + c)
print(c.shape)
print(img_fft[:,0]-c)
print(img_fft[:,0])

#%%
plot = True
if plot:
    nice_imshow(torch.fft.fftshift(torch.fft.fft(img, norm = 'forward')).abs())
    nice_imshow(torch.fft.fftshift(torch.fft.fft(v_eval, norm = 'forward')).abs())
    nice_imshow(torch.fft.fftshift(torch.fft.fft(img, norm = 'forward')).real)
    nice_imshow(torch.fft.fftshift(torch.fft.fft(v_eval, norm = 'forward')).real)
    nice_imshow(torch.fft.fftshift(torch.fft.fft(img, norm = 'forward')).imag)
    nice_imshow(torch.fft.fftshift(torch.fft.fft(v_eval, norm = 'forward')).imag)
    #nice_imshow(v_eval.real)
    plt.show()

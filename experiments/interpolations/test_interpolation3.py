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
img = torch.rand(2*N + 1)
coeffs = torch.fft.fftshift(torch.fft.fft(img, norm = 'forward'))
coeffs.imag[[0,-1]] = 0.
img = torch.fft.ifft(torch.fft.ifftshift(coeffs))

def trigo_val1d(x, coeffs):
    N = coeffs.shape[0]//2
    res = 0
    for k in range(2*N + 1):
            res += coeffs[k]*torch.exp(1j * 2 * np.pi * ((k-N) * x))
    return res

s = torch.tensor(np.linspace(0,1, 2*N+1, endpoint=False))
v_eval = trigo_val1d(s, coeffs)
v_eval_fft = torch.fft.fftshift(torch.fft.fft(v_eval, norm = 'forward'))
img_fft = torch.fft.fftshift(torch.fft.fft(img, norm='forward'))
print(v_eval_fft)
print(img_fft)
print(coeffs)
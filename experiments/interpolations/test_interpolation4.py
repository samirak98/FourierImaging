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

N = 3
img = torch.rand(2*N,2*N)
coeffs = torch.fft.fftshift(torch.fft.fft2(img, norm = 'forward'))
first_row = coeffs[:,1]
first_col = coeffs[1,:]

nice_imshow(coeffs.abs())
plt.show()

print(first_row)
print(first_col)
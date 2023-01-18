#%%
import torch
import torch.nn as nn
from torchvision import transforms
import yaml
import matplotlib.pyplot as plt

#%% custom imports
import sys, os
sys.path.append(os.path.abspath('../../'))

import fourierimaging as fi

from fourierimaging.utils.helpers import load_model, init_opt, fix_seed
from fourierimaging.modules import TrigonometricResize_2d, SpectralConv2d,\
                                   spectral_to_spatial, conv_to_spectral
from fourierimaging.utils import datasets as data
import fourierimaging.train as train
import numpy as np

#%%
in_channels = 3
out_channels = 7
im_shape = [17,19]

conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,\
                 kernel_size=1, padding=0, padding_mode='circular' ,bias=False)
sp_conv = conv_to_spectral(conv, im_shape, parametrization='spectral')
spp_conv = conv_to_spectral(conv, im_shape, parametrization='spatial')
#%%
x = torch.rand(size=(1,in_channels,im_shape[0], im_shape[1]))
cx = conv(x)
scx = sp_conv(x)
sccx = spp_conv(x)

print(torch.max(scx-cx).item())
print(torch.max(scx-sccx).item())

#%%
import torch
import torch.nn as nn
from torchvision import transforms
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('webagg')
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
in_channels = 1
out_channels = 1
n = 7

conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,\
                 kernel_size=n, padding='same', padding_mode='circular' ,bias=False)

sp_conv = conv_to_spectral(conv, [n,n], parametrization='spectral')


x = torch.rand(1,1,n,n)

eval_1 = torch.sum(conv(x))
eval_2 = torch.sum(sp_conv(x))

eval_1.backward()
eval_2.backward()

g_1 = conv.weight.grad
g_2 = sp_conv.weight.grad

fg_2 = spectral_to_spatial(g_2, [n,n])

print(g_1/fg_2)
print(torch.max(g_1-fg_2))





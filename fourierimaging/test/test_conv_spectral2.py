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
conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,\
                 kernel_size=1, padding=0, padding_mode='circular' ,bias=False)
max_err = 0.
for i in [27,28]:
    for j in [27, 28]:
        im_shape = [i,j]
        print(im_shape)
        sp_conv = conv_to_spectral(conv, im_shape, parametrization='spectral')
        spp_conv = conv_to_spectral(conv, im_shape, parametrization='spatial')
        
        for t in range(10):
            x = torch.rand(size=(1,in_channels,im_shape[0], im_shape[1]))
            cx = conv(x)
            scx = sp_conv(x)
            sccx = spp_conv(x)

            sc_err  = torch.max(scx-cx).item()
            scc_err = torch.max(sccx-cx).item()

        if max(sc_err, scc_err) > 1e-4:
            print('The error seems to high, max error: ' +str(sc_err) + str(scc_err) + ' at (i,j): '\
                    + str((i,j)))

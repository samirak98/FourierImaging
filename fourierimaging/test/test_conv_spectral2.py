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
in_channels = 3
out_channels = 7
conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,\
                 kernel_size=19, padding=9, padding_mode='circular' ,bias=False)
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

            sc_err  = torch.max(torch.abs(scx-cx)).item()
            scc_err = torch.max(torch.abs(sccx-cx)).item()
            #scc_err = 0.

        if max(sc_err, scc_err) > 1e-5:
            print('The error seems too high')
            print('Spectral Error: ' + str(sc_err))
            print('Spatial Error: '   + str(scc_err))
            print(' at (i,j): ' + str((i,j)))
        else:
            print('Cool :)')


#%%
print(50*'.')
print('Running Test 2')

plt.close('all')
for i in [5,6]:
    for j in [5, 6]:
        im_shape = [i,j]
        print(im_shape)
        sp_conv = conv_to_spectral(conv, im_shape, parametrization='spectral', conv_like_cnn=True)
        odd = ((im_shape[-1]%2) == 1)
        w = spectral_to_spatial(sp_conv.weight.data, im_shape, odd=odd, conv_like_cnn=False)
        spp_conv =  SpectralConv2d(in_channels, out_channels, w, parametrization='spatial', odd=odd, conv_like_cnn=False)
        x = torch.rand(size=(1,in_channels,im_shape[0], im_shape[1]))
        cx = conv(x)
        scx = sp_conv(x)
        sccx = spp_conv(x)
        spec_spat_err = torch.max(torch.abs(sccx-scx)).item()
        spec_conv_err = torch.max(torch.abs(scx-cx)).item()
        spat_conv_err = torch.max(torch.abs(cx-sccx)).item()

        plotting = False
        if plotting:
            plt.figure()
            plt.imshow(cx.detach().numpy()[0,0])
            plt.colorbar()
            plt.figure()
            plt.imshow(scx.detach().numpy()[0,0])
            plt.colorbar()
            plt.figure()
            plt.imshow(sccx.detach().numpy()[0,0])
            plt.colorbar()
            plt.show()
        
        if max(spec_spat_err, spec_conv_err, spat_conv_err) > 1e-5:
            #print('The error seems too high')
            print('Spectral - Spatial: ' + str(spec_spat_err))
            print('spectralFNO - CONV: ' + str(spec_conv_err))
            print('spatialFNO - CONV: ' + str(spat_conv_err))
            print(' at (i,j): ' + str((i,j)))
        else:
            print('Cool :)')

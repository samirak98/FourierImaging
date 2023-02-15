#%%
import torch
from torchvision import transforms
import yaml

#%% custom imports
import sys, os
sys.path.append(os.path.abspath('../../'))

import fourierimaging as fi

from fourierimaging.utils.helpers import load_model, init_opt, fix_seed
from fourierimaging.modules import TrigonometricResize_2d
from fourierimaging.utils import datasets as data
import fourierimaging.train as train
import numpy as np

#%%
diff = 0
for i in range(100,145):
    add_y = np.random.randint(10,20)
    size = [i, i+add_y]
    add_x = np.random.randint(10,20)
    add_y = np.random.randint(10,20)
    #add_y = add_x
    size_new = [size[0]+add_x, size[1]+add_y]
    sizing_new = TrigonometricResize_2d(size_new)
    sizing_re = TrigonometricResize_2d(size)

    dtypes = [torch.float, torch.cfloat]
    for dtype in dtypes:
        x = torch.rand(size=size, dtype=dtype)
        x_new = sizing_new(x)
        re_x = sizing_re(x_new)

        loc_diff = torch.norm(x-re_x, p=float('inf'))
        print('Sizing from '+str(x.shape)+ ' to ' + str(x_new.shape)\
            + ' yields an error: ' + str(loc_diff) + ' for dtype '\
            + str(dtype))

        diff = max(loc_diff, diff)

print('The total max difference is' + str(diff))

#%%
a = torch.rand(7,7)
arft = torch.fft.rfft2(a)
print(arft.shape)
ainter = torch.fft.irfft2(arft, s=(7,6))
ainterrft = torch.fft.rfft2(ainter)
ainterinter = torch.fft.irfft2(arft, s=(7,6))
print(ainterrft.shape)
print(torch.norm(arft - ainterrft))

# %% define img
img = torch.rand(1,1,8,8)
resize = torch.nn.functional.interpolate
inter_img = resize(img, (4,4), mode='bilinear', align_corners=False)
print(img)
print(inter_img)
# %%
a = torch.rand(5,5)
a_ishift = torch.fft.ifftshift(a)

print(a)
print(a_ishift)
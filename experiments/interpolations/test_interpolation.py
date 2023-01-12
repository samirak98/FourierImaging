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
for i in range(100,135):
    size = [i,i+3]
    add_x = np.random.randint(10,20)
    add_y = np.random.randint(10,20)
    size_new = [i+add_x, i+add_y]
    sizing_new = TrigonometricResize_2d(size_new)
    sizing_re = TrigonometricResize_2d(size)

    x = torch.rand(size=size, dtype=torch.cfloat)

    x_new = sizing_new(x)
    re_x = sizing_re(x_new)

    loc_diff = torch.norm(x-re_x, p=float('inf'))
    print('Sizing from '+str(x.shape)+ ' to ' + str(x_new.shape) + ' yields an error: ' + str(loc_diff))

    diff = max(loc_diff, diff)
print(diff)
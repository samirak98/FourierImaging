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
for i in range(100,105):
    size = [i,i]
    add = np.random.randint(10)
    print(add)
    new_size = [i+add,i+add]
    new_sizing = TrigonometricResize_2d(new_size)
    re_sizing = TrigonometricResize_2d(size)

    x = torch.rand(size=size)

    new_x = new_sizing(x)
    re_x = re_sizing(new_x)

    diff = max(torch.norm(x-re_x, p=float('inf')), diff)
print(diff)
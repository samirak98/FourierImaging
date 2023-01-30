import torch
import torch.nn as nn
from torchvision import transforms
import yaml
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl 
#mpl.use('webagg')
from omegaconf import DictConfig, OmegaConf, open_dict

# custom imports
#%% custom imports
import sys, os
sys.path.append(os.path.abspath('../../'))

import fourierimaging as fi

from fourierimaging.utils.helpers import load_model, init_opt, fix_seed
from fourierimaging.modules import TrigonometricResize_2d, conv_to_spectral,\
                                   SpectralConv2d, irfftshift
from fourierimaging.modules import SpectralCNN, SpectralResNet
from fourierimaging.utils import datasets as data
import fourierimaging.train as train

#%%
device = 'cuda:1'
#path = '../saved_models/resnet-20230129-153250'
path = '../saved_models/resnet-20230130-061405'

conf = torch.load(path, map_location=torch.device(device))['conf']
with open_dict(conf):
    conf['dataset']['path'] = '../../../datasets'
    conf.train.device =device
train_loader, valid_loader, test_loader = data.load(conf.dataset)

conf = torch.load(path, map_location=torch.device(device))['conf']
with open_dict(conf):
    conf.train.device = device
history = torch.load(path, map_location=torch.device(device))['history']

tester = train.Tester(test_loader, conf.train)

model = load_model(conf).to(device)
model.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])

spectral_model = SpectralResNet.from_resnet(model, [112, 112], stride_trigo=True).to(device)

#%%
tester = train.Tester(test_loader, conf.train)
#tester(model)
#print(model)
print(spectral_model)
total_params = sum(p.numel() for p in spectral_model.parameters())
print('Number of params: ' + str(total_params))

x,y = next(iter(test_loader))
x = x.to(device)

z = model.conv1(x)
z = model.bn1(z)
z = model.relu(z)
z = model.maxpool(z)
z = model.layer1(z)
z = model.layer2(z)
z = model.layer3(z)
#z = model.layer4(z)

sz = spectral_model.conv1(x)
sz = spectral_model.bn1(sz)
sz = spectral_model.relu(sz)
sz = spectral_model.maxpool(sz)
sz = spectral_model.layer1(sz)
sz = spectral_model.layer2(sz)
sz = spectral_model.layer3(sz)
#sz = spectral_model.layer4(sz)

print(torch.linalg.norm(z-sz))
print(torch.max(model(x) - spectral_model(x)))
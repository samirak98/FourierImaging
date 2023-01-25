import torch
import torch.nn as nn
from torchvision import transforms
import yaml
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl 
mpl.use('webagg')

# custom imports
#%% custom imports
import sys, os
sys.path.append(os.path.abspath('../../'))

import fourierimaging as fi

from fourierimaging.utils.helpers import load_model, init_opt, fix_seed
from fourierimaging.modules import TrigonometricResize_2d, conv_to_spectral, SpectralConv2d, spectral_to_spatial
from fourierimaging.modules import SpectralCNN
from fourierimaging.utils import datasets as data
import fourierimaging.train as train


#%% Set up variable and data for an example
experiment_file = '../classification/FMNIST.yaml'
with open(experiment_file) as exp_file:
    conf = yaml.safe_load(exp_file)

conf['dataset']['path'] = '../../../datasets'
#%% fix random seed
fix_seed(conf['seed'])

#%% get train, validation and test loader
train_loader, valid_loader, test_loader = data.load(conf['dataset'])

#%% define the model
if conf['CUDA']['use_cuda'] and torch.cuda.is_available():
    device = torch.device("cuda" + ":" + str(conf['CUDA']['cuda_device']))
else:
    device = "cpu"
conf['train']['device'] = device
model = load_model(conf).to(device)
path = '../saved_models/simple_cnn-spectral-spectral-14-small'
model.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])

w = model.layers1.conv.weight
wsp = spectral_to_spatial(w, [28,28], odd=False)

wsp = wsp.detach().to('cpu').numpy()

n= 2

fig, ax = plt.subplots(n,n)
for i in range(n):
    for j in range(n):
        ax[i,j].imshow(np.abs(wsp[0,i*n+j,:,:]))
#plt.show()

fig, ax = plt.subplots(1,1)
ax.imshow(np.abs(wsp[0,0,:,:]))
plt.show()
#%%
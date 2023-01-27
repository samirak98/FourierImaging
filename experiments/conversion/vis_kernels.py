import torch
import yaml
import time

# custom imports
#%% custom imports
import sys, os
sys.path.append(os.path.abspath('../../'))

import fourierimaging as fi

from fourierimaging.utils.helpers import load_model, init_opt, fix_seed
from fourierimaging.utils import datasets as data
from fourierimaging.modules import TrigonometricResize_2d, conv_to_spectral,\
                                   SpectralConv2d, irfftshift
import fourierimaging.train as train

import matplotlib.pyplot as plt

#%% Set up variable and data for an example
experiment_file = '../classification/FMNIST.yaml'
with open(experiment_file) as exp_file:
    conf = yaml.safe_load(exp_file)

#%% fix random seed
fix_seed(conf['seed'])

#%% get train, validation and test loader
conf['dataset']['path'] = '../../../datasets'
train_loader, valid_loader, test_loader = data.load(conf['dataset'])

#%% define the model
if conf['CUDA']['use_cuda'] and torch.cuda.is_available():
    device = torch.device("cuda" + ":" + str(conf['CUDA']['cuda_device']))
else:
    device = "cpu"
conf['train']['device'] = device
model = load_model(conf).to(device)

path = '../saved_models/simple_cnn-spectral'
model.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])

#%%
i = 0
j = 1
w = model.layers1.conv.weight
w = irfftshift(w)
w = torch.fft.fftshift(torch.fft.irfft2(w), dim=[-2,-1]).real
w = w[i,j,:,:].detach().numpy()

plt.imshow(w)
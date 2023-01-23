import torch
import torch.nn as nn
from torchvision import transforms
import yaml
import numpy as np
import csv

# custom imports
#%% custom imports
import sys, os
sys.path.append(os.path.abspath('../../'))

import fourierimaging as fi

from fourierimaging.utils.helpers import load_model, init_opt, fix_seed
from fourierimaging.modules import TrigonometricResize_2d, conv_to_spectral, SpectralConv2d
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
path = '../saved_models/simple_cnn-circular'
model.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])

#%%
spectral_model = SpectralCNN(model).to(device)

tester = train.Tester(test_loader, conf['train'])
tester(model)
tester(spectral_model)
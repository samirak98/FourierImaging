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
from fourierimaging.modules import SpectralCNN
from fourierimaging.utils import datasets as data
import fourierimaging.train as train

#%%
paths = ['../saved_models/cnns/cnn-3-3',
         '../saved_models/cnns/cnn-5-5',
         #'../saved_models/cnns/spectral-cnn-spatial-3-3-20230126-173803',
         #'../saved_models/cnns/spectral-cnn-spatial-5-5-20230126-175357'
         '../saved_models/cnns/spectral-cnn-spatial-3-3',
         '../saved_models/cnns/spectral-cnn-spatial-5-5'
        ]

fig, ax = plt.subplots()
for path in paths:
    conf = torch.load(path, map_location=torch.device('cpu'))['conf']
    history = torch.load(path, map_location=torch.device('cpu'))['history']


    #model.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])

    with open_dict(conf):
        conf['dataset']['path'] = '../../../datasets'
    if conf['CUDA']['use_cuda'] and torch.cuda.is_available():
        device = torch.device("cuda" + ":" + str(conf['CUDA']['cuda_device']))
    else:
        device = "cpu"
    #%% fix random seed
    fix_seed(conf['seed'])

    #%% get train, validation and test loader
    train_loader, valid_loader, test_loader = data.load(conf['dataset'])

    ax.plot(history['val_acc'], label=path)

    #%% define the model

    #model = load_model(conf).to(device)
ax.legend()
plt.show()
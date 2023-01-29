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

def main():
    accs = []
    paths = [
             '../saved_models/cnns/spectral-cnn-spectral-3-3',
            ]

    conf = torch.load(paths[0], map_location=torch.device('cpu'))['conf']
    with open_dict(conf):
        conf['dataset']['path'] = '../../../datasets'
        conf.train.device ='cpu'
    train_loader, valid_loader, test_loader = data.load(conf.dataset)

    for path in paths:
        print(50*':')
        print(path)
        conf = torch.load(path, map_location=torch.device('cpu'))['conf']
        with open_dict(conf):
            conf.train.device ='cpu'
        history = torch.load(path, map_location=torch.device('cpu'))['history']
    
        plt.plot(history['val_acc'])

    
        #%% define the model
    
        #model = load_model(conf).to(device)

if __name__ == '__main__':
    main()
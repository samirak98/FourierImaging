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
from matplotlib import rc
from cycler import cycler

#%%
default_cycler = (cycler(color=['xkcd:sky', 'xkcd:grapefruit',
                                'steelblue','tab:pink',
                                'olive','xkcd:apple','xkcd:grapefruit',
                                'xkcd:muted blue','peru','tab:pink',
                                'deeppink', 'steelblue', 'tan', 'sienna',
                                'olive', 'coral']))

plt.style.use(['seaborn-whitegrid'])
rc('font',**{'family':'lmodern','serif':['Times'],'size':10})
rc('text', usetex=True)
rc('lines', linewidth=2, linestyle='-')
rc('axes', prop_cycle=default_cycler)

def main():
    plt.close('all')
    fig, ax = plt.subplots(1,2, sharey=True, figsize=(8.27/1.5,11.69/5))
    
    accs = []
    paths = {
             'FNO 3 x 3' : '../saved_models/cnns/spectral-cnn-spectral-3-3',
             'CNN 3 x 3': '../saved_models/cnns/cnn-3-3',
             'FNO 15 x 15':'../saved_models/cnns/spectral-cnn-spectral-15-15',
             'CNN 15 x 15' : '../saved_models/cnns/cnn-15-15',
            }

    conf = torch.load(paths['FNO 3 x 3'], map_location=torch.device('cpu'))['conf']
    with open_dict(conf):
        conf['dataset']['path'] = '../../../datasets'
        conf.train.device ='cpu'
    #train_loader, valid_loader, test_loader = data.load(conf.dataset)

    for name in paths.keys():
        print(50*':')
        path = paths[name]
        print(path)
        conf = torch.load(path, map_location=torch.device('cpu'))['conf']
        with open_dict(conf):
            conf.train.device ='cpu'
        history = torch.load(path, map_location=torch.device('cpu'))['history']
        
       
        ax[0].plot(history['train_acc'], label=name)
        ax[1].plot(history['val_acc'], label=name)

    ax[0].legend()
    #ax[1].legend()
    ax[0].set_xlabel('Epoch')
    ax[1].set_xlabel('Epoch')
    ax[0].set_ylabel('Train Accuracy')
    ax[1].set_ylabel('Test Accuracy')
    
    save=True
    if save:
        plt.tight_layout(pad=0.2)
        plt.savefig('train.pdf')
        #%% define the model
    
        #model = load_model(conf).to(device)

if __name__ == '__main__':
    main()
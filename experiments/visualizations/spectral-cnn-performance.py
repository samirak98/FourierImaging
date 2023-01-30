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

accs = []
paths = [
            '../saved_models/cnns/cnn-3-3',
            '../saved_models/cnns/cnn-5-5',
            '../saved_models/cnns/cnn-15-15',
            '../saved_models/cnns/cnn-20-20',
            '../saved_models/cnns/cnn-25-25',
            '../saved_models/cnns/cnn-28-28',
            '../saved_models/cnns/spectral-cnn-spectral-3-3',
            '../saved_models/cnns/spectral-cnn-spectral-5-5',
            '../saved_models/cnns/spectral-cnn-spectral-10-10',
            '../saved_models/cnns/spectral-cnn-spectral-15-15',
            '../saved_models/cnns/spectral-cnn-spectral-20-20',
            '../saved_models/cnns/spectral-cnn-spectral-25-25',
            '../saved_models/cnns/spectral-cnn-spectral-28-28'
        ]

conf = torch.load(paths[0])['conf']
with open_dict(conf):
    conf['dataset']['path'] = '../../../datasets'
    #conf.train.device ='cpu'

device = conf.train.device
train_loader, valid_loader, test_loader = data.load(conf.dataset)

#path = '../saved_models/cnns/cnn-28-28'


paths1 = ['../saved_models/cnns/cnn-3-3', '../saved_models/cnns/cnn-28-28']
for path in paths1:
    conf = torch.load(path, map_location=torch.device(device))['conf']
    history = torch.load(path, map_location=torch.device(device))['history']
    tester = train.Tester(test_loader, conf.train)
    model = load_model(conf).to(device)
    model.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])

    for s in [3,5,10,15,20,25,28]:
        spectral_model = SpectralCNN.from_CNN(model, im_shape=[28,28], ksize=[s,s])
        #print(spectral_model)
        acc = tester(spectral_model)['test_acc']
        accs.append([path + 'to-spetral-' + str(s) + '-' + str(s), acc])

for path in paths:
    print(50*':')
    print(path)
    conf = torch.load(path, map_location=torch.device(device))['conf']
    with open_dict(conf):
        conf.train.device =device
    history = torch.load(path, map_location=torch.device(device))['history']

    tester = train.Tester(test_loader, conf.train)
    
    model = load_model(conf).to(device)
    model.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])
    #print(model)
    
    acc = tester(model)['test_acc']
    accs.append([path, acc])
    



fname = 'spectral-cnn-perf'
with open(fname, 'w') as f:
    writer = csv.writer(f, lineterminator = '\n')
    for i in range(len(accs)):
        writer.writerow(accs[i])


    #%% define the model

    #model = load_model(conf).to(device)

import torch
from torchvision import transforms
import yaml
import numpy as np
import csv
from tqdm.auto import tqdm

# custom imports
#%% custom imports
import sys, os
sys.path.append(os.path.abspath('../../'))

import fourierimaging as fi

from fourierimaging.utils.helpers import load_model, init_opt, fix_seed
from fourierimaging.modules import TrigonometricResize_2d, SpectralCNN, SpectralResNet
from fourierimaging.utils import datasets as data
import fourierimaging.train as train
import fourierimaging.adversarial as adversarial

from omegaconf import DictConfig, OmegaConf, open_dict

path = '../saved_models/cnns/cnn-5-5'
#path = '../saved_models/cnn-5-5-20230129-223101'
conf = torch.load(path)['conf']
spectral = True

with open_dict(conf):
    conf['dataset']['path'] = '../../../datasets'

device = conf.train.device
#%% get train, validation and test loader
train_loader, valid_loader, test_loader = data.load(conf['dataset'])

#%% load model
model = load_model(conf).to(device)
model.load_state_dict(torch.load(path)['model_state_dict'])

#%% define adversarial attack
loss = train.select_loss(conf['train']['loss'])
#attack = adversarial.fgsm(loss, epsilon=0.1, x_min=0.0, x_max=1.0)
attack = adversarial.pgd(loss, epsilon=None, x_min=0.0, x_max=1.0, restarts=1, 
                         attack_iters=7, alpha=None, alpha_mul=1.0, norm_type="l2")

#%% define Tester
tester = train.Tester(test_loader, conf['train'], attack = attack)

#%% eval
tester(model)
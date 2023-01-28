import torch
import yaml
import time

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra

from hydra import compose, initialize
from omegaconf import OmegaConf

# custom imports
#%% custom imports
from inspect import getsourcefile
import os.path as path, sys
import sys, os
sys.path.append(os.path.abspath('../../'))

#import fourierimaging as fi

import fourierimaging as fi

from fourierimaging.utils.helpers import load_model, init_opt, fix_seed
from fourierimaging.modules import TrigonometricResize_2d, SpectralConv2d, spectral_to_spatial
from fourierimaging.utils import datasets as data
import fourierimaging.train as train
import numpy as np
initialize(config_path="../../experiments/conf")
conf = compose(config_name="config", overrides=["+model=spectralresnet", "+dataset=CUB200"])
print(OmegaConf.to_yaml(conf))

#%% fix random seed
fix_seed(conf.seed)


#%% define the model
if conf.CUDA.use_cuda and torch.cuda.is_available():
    device = torch.device("cuda" + ":" + str(conf.CUDA.cuda_device))
else:
    device = "cpu"

with open_dict(conf):
    conf.train['device'] = str(device)

model = load_model(conf).to(device)

#%% Initialize optimizer and lamda scheduler
opt, lr_scheduler = init_opt(model, conf['train']['opt'])
# initalize history
tracked = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
history = {key: [] for key in tracked}
total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
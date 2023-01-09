import torch
import torch.nn as nn
from modules.perceptron import perceptron
import random
import numpy as np
#%% set a fixed seed
def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#%% load model by type
def load_model(conf):
    model_conf = conf['model']
    # load activation function
    if model_conf['activation_function'] == 'ReLU':
        act_fun = nn.ReLU
    
    if model_conf['type'] == 'perceptron':
        model = perceptron(model_conf['sizes'], act_fun = act_fun,\
                           mean=conf['dataset']['mean'], std=conf['dataset']['std'])
    else:
        raise ValueError('Unknown model type: ' + conf['type'])
    
    return model

#%% initialize optimizer
def init_opt(model, conf):
    if conf['name'] == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr = conf['lr'], momentum = conf['momentum'])
    else:
        raise ValueError('Unknown optimizer: ' + conf['name'])
    return opt
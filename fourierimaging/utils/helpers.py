import torch
import torch.nn as nn
from ..modules.perceptron import perceptron
from ..modules.simple_cnn import simple_cnn
from ..modules.resnet import load_resnet
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
    if 'activation_function' in model_conf:
        if model_conf['activation_function'] == 'ReLU':
            act_fun = nn.ReLU
        else:
            raise Warning('Unknown activation function: ' + model_conf['activation_function'] +\
                            ' specified, using ReLU instead.')
            act_fun = nn.ReLU
                
    
    if model_conf['type'] == 'perceptron':
        model = perceptron(model_conf['sizes'], act_fun = act_fun,\
                           mean=conf['dataset']['mean'], std=conf['dataset']['std'])
    elif model_conf['type'] == 'simple_cnn':
        model = simple_cnn(act_fun = act_fun,\
                           mean=conf['dataset']['mean'], std=conf['dataset']['std'])
    elif model_conf['type'] == 'resnet':
        
        model = load_resnet(size=model_conf['size'],\
                            mean=conf['dataset']['mean'], std=conf['dataset']['std'])
    else:
        raise ValueError('Unknown model type: ' + model_conf['type'])
    
    return model

#%% initialize optimizer
def init_opt(model, conf):
    if conf['name'] == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr = conf['lr'], momentum = conf['momentum'])
    elif conf['name'] == 'Adam':
        opt = torch.optim.Adam(model.parameters())
    else:
        raise ValueError('Unknown optimizer: ' + conf['name'])
    return opt
import torch
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
    if conf['type'] == 'perceptron':
        model = perceptron(conf['sizes'], conf['activation_function'])
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
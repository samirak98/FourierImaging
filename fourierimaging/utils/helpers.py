import torch
import torch.nn as nn
from torchvision import models
from ..modules.perceptron import perceptron
from ..modules.simple_cnn import CNN, SpectralCNN
from ..modules.resnet import resnet18
import random
import numpy as np
#%% set a fixed seed
def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
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
        model = CNN(mean=conf['dataset']['mean'], std=conf['dataset']['std'])
        if model_conf['spectral']['use']:
            if model_conf['spectral']['cnn-init']:
                model = SpectralCNN.from_CNN(model, fix_out = True, parametrization=model_conf['spectral']['parametrization'])
            else:
                model = SpectralCNN(fix_out = True, parametrization=model_conf['spectral']['parametrization'])

    elif model_conf['type'] == 'resnet':
        if model_conf['pretrained']:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            model = resnet18(padding_mode=model_conf['padding_mode'])
        # model = load_resnet(size=model_conf['size'],\
        #                     mean=conf['dataset']['mean'], std=conf['dataset']['std'], num_classes=conf['dataset']['num_classes'])
    elif model_conf['type'] == 'efficentnet':
        model = models.efficientnet_b1(pretrained=model_conf['pretrained'], weights=models.EfficientNet_B1_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=conf['dataset']['num_classes'])
    else:
        raise ValueError('Unknown model type: ' + model_conf['type'])
    
    return model

#%% initialize optimizer
def init_opt(model, conf):
    if conf['name'] == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr = conf['lr'], momentum = conf['momentum'])
    elif conf['name'] == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=conf['lr'])
    else:
        raise ValueError('Unknown optimizer: ' + conf['name'])
        
    lr_scheduler = None
    if 'lr_scheduler' in conf:
        if conf['lr_scheduler'] == 'Plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.7, patience=5,threshold=0.01)
        
    
    return opt, lr_scheduler
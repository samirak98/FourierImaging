import torch
import torch.nn as nn
from torchvision import models
from ..modules.perceptron import perceptron
from ..modules.cnn import CNN
from ..modules.spectralcnn import SpectralCNN
from ..modules.resnet import resnet18
from ..modules.spectralresnet import spectralresnet18
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
    model_conf = conf.model
    
    # load activation function
    if 'activation_function' in model_conf:
        if model_conf.activation_function == 'ReLU':
            act_fun = nn.ReLU
        else:
            raise Warning('Unknown activation function: ' + model_conf.activation_function +\
                            ' specified, using ReLU instead.')
            act_fun = nn.ReLU
    if model_conf.type == 'perceptron':
        model = perceptron(model_conf.sizes, act_fun = act_fun,\
                           mean=conf.dataset.mean, std=conf.dataset.std)
    elif model_conf.type == 'simple_cnn':
        if 'stride' in model_conf.keys():
            stride = model_conf.stride
        else:
            stride = 2

        model = CNN(mean=conf.dataset.mean, std=conf.dataset.std,\
                    ksize1=model_conf.ksize[0], ksize2 = model_conf.ksize[1],\
                    mid_channels=model_conf.mid_channels,\
                    out_channels=model_conf.out_channels,
                    stride = stride)
        if model_conf.spectral.use:
            if model_conf.spectral.cnninit:
                model = SpectralCNN.from_CNN(
                            model, fix_out = True,\
                            parametrization=model_conf.spectral.parametrization,
                            norm=model_conf.spectral.norm,
                            conv_like_cnn = model_conf.spectral.conv_like_cnn
                        )
            else:
                model = SpectralCNN(
                            mean=conf.dataset.mean, std=conf.dataset.std,\
                            ksize1=model_conf.ksize[0], ksize2 = model_conf.ksize[1],\
                            mid_channels=model_conf.mid_channels,\
                            out_channels=model_conf.out_channels,\
                            fix_in = True,
                            fix_out = True, 
                            parametrization=model_conf.spectral.parametrization,
                            norm=model_conf.spectral.norm,
                            conv_like_cnn = model_conf.spectral.conv_like_cnn,
                            stride = stride
                        )

    elif model_conf.type == 'resnet':
        if not model_conf.spectral.use:
            if model_conf.pretrained:
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                model = resnet18(padding_mode=model_conf.padding_mode, 
                                 num_classes=conf.dataset.num_classes,
                                 stride_trigo = model_conf.stride_trigo)
        else:
            model = spectralresnet18()


    elif model_conf.type == 'efficentnet':
        model = models.efficientnet_b1(pretrained=model_conf.pretrained, weights=models.EfficientNet_B1_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=conf.dataset.num_classes)
    else:
        raise ValueError('Unknown model type: ' + model_conf.type)
    
    return model

#%% initialize optimizer
def init_opt(model, conf):
    if conf.name == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr = conf.lr, momentum = conf.momentum)
    elif conf.name == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=conf.lr)
    else:
        raise ValueError('Unknown optimizer: ' + conf.name)
        
    lr_scheduler = None
    if 'lr_scheduler' in conf:
        if conf.lr_scheduler == 'Plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.7, patience=5,threshold=0.01)
        
    
    return opt, lr_scheduler
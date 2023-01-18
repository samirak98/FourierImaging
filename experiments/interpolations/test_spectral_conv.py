#%%
import torch
from torchvision import transforms
import yaml
import matplotlib.pyplot as plt

#%% custom imports
import sys, os
sys.path.append(os.path.abspath('../../'))

import fourierimaging as fi

from fourierimaging.utils.helpers import load_model, init_opt, fix_seed
from fourierimaging.modules import TrigonometricResize_2d, SpectralConv2d, spectral_to_spatial
from fourierimaging.utils import datasets as data
import fourierimaging.train as train
import numpy as np

#%%
def show_img(x, b=0, c=0):
    z = x.detach().numpy()
    z = z[b,c,:,:]
    z = np.abs(z)
    plt.imshow(z)

#%% Set up variable and data for an example
experiment_file = '../classification/MNIST.yaml'
with open(experiment_file) as exp_file:
    conf = yaml.safe_load(exp_file)

conf['dataset']['path'] = '../../../datasets'
#%% fix random seed
fix_seed(conf['seed'])

#%% get train, validation and test loader
train_loader, valid_loader, test_loader = data.load(conf['dataset'])

#%% define the model
if conf['CUDA']['use_cuda'] and torch.cuda.is_available():
    device = torch.device("cuda" + ":" + str(conf['CUDA']['cuda_device']))
else:
    device = "cpu"
conf['train']['device'] = device

#%% Define model
out_channels = 20
out_shape = (7,56)
model = SpectralConv2d(1, out_channels, parametrization='spectral',out_shape=out_shape)

model_spatial = SpectralConv2d(1, out_channels,\
                               parametrization='spatial',\
                               out_shape=out_shape,
                               ksize1=28, ksize2=28)
model_spatial.weights.data = spectral_to_spatial(model.weights, [28, 28])

w = model.weights
w_spatial = model_spatial.weights

print(w[0,0,:,:] - w_spatial[0,0,14,14])
x = torch.rand(size=[1,1,28,28])

mx = model(x)
msx = model_spatial(x)



#%%
def main():
    
    out_channels = 20
    out_shape = (7,56)
    model = SpectralConv2d(1, out_channels,\
                           parametrization='spectral',out_shape=out_shape,\
                           modes1=1, modes2=1)
    
    model_spatial = SpectralConv2d(1, out_channels,\
                                   parametrization='spatial',\
                                   out_shape=out_shape,
                                   ksize1=28, ksize2=28)
    model_spatial.weights.data = model.convert([28, 28])
    # for i, (x,y) in enumerate(train_loader):
    #     xx=model(x)
    #     xxx = model_spatial(x)
    #     print(torch.max(xx-xxx).item())
        
if __name__=='__main__':
    main()



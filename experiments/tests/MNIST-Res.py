import torch
import yaml

import os, sys
import os.path
sys.path.append('../utils/')
# custom imports
from utils import helper
from utils import datasets as data
import train

#%% Set up variable and data for an example
experiment_file = 'utils/experiments/classification/MNIST.yaml'
with open(experiment_file) as exp_file:
    conf = yaml.safe_load(exp_file)

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
model = load_model(conf).to(device)
path = '../saved_model/perceptron-MNIST'
model.load_state_dict(torch.load(path))
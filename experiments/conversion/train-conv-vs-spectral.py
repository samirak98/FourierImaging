import torch
import yaml
import time

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra

# custom imports
#%% custom imports
import sys, os
sys.path.append(os.path.abspath('../../'))

import fourierimaging as fi

from fourierimaging.utils.helpers import load_model, init_opt, fix_seed
from fourierimaging.utils import datasets as data
from fourierimaging.modules import SpectralCNN
import fourierimaging.train as train


path = '../saved_models/cnns/cnn-28-28'
#path = '../saved_models/cnn-5-5-20230129-223101'
conf = torch.load(path)['conf']

with open_dict(conf):
    conf['dataset']['path'] = '../../../datasets'

#%% fix random seed
fix_seed(conf.seed)

#%% get train, validation and test loader
train_loader, valid_loader, test_loader = data.load(conf.dataset)

#%% define the model
if conf.CUDA.use_cuda and torch.cuda.is_available():
    device = torch.device("cuda" + ":" + str(conf.CUDA.cuda_device))
else:
    device = "cpu"

with open_dict(conf):
    conf.train['device'] = str(device)

model = load_model(conf).to(device)
spectral_model = SpectralCNN.from_CNN(model)

#%% Initialize optimizer and lamda scheduler
#opt, lr_scheduler = init_opt(model, conf['train']['opt'])
#
opt = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.)
sp_opt = torch.optim.SGD(spectral_model.parameters(), lr = 0.01, momentum = 0.)
lr_scheduler = None
# initalize history
tracked = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
history = {key: [] for key in tracked}
trainer = train.trainer(model, opt, lr_scheduler, train_loader, valid_loader, conf['train'])
sp_trainer = train.trainer(spectral_model, sp_opt, lr_scheduler, train_loader, valid_loader, conf['train'])
total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(OmegaConf.to_yaml(conf))
#%%
print(50*'#')
print('Starting training.')
print('Model name ' + model.name())
print('Total number of params: ' + str(total_params) + ' parameters')
print('Number of trainable params: ' + str(total_trainable_params))

for i in range(conf['train']['epochs']):
    print(50*".")
    print('Starting epoch: ' + str(i))
    print('Learning rate: ' + str(opt.param_groups[0]['lr']))
    print(50*"=")
    
    # train_step
    train_data = trainer.train_step()
    train_data = sp_trainer.train_step()
    
    # validation step
    val_data = trainer.validation_step()
    

tester = train.Tester(test_loader, conf.train)
tester(model)


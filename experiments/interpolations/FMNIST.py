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

from omegaconf import DictConfig, OmegaConf, open_dict
from select_sizing import sizing

path = '../saved_models/cnns/cnn-5-5'
#path = '../saved_models/cnn-5-5-20230129-223101'
conf = torch.load(path)['conf']
spectral = True

with open_dict(conf):
    conf['dataset']['path'] = '../../../datasets'

device = conf.train.device
#%% get train, validation and test loader
train_loader, valid_loader, test_loader = data.load(conf['dataset'])


model = load_model(conf).to(device)

model.load_state_dict(torch.load(path)['model_state_dict'])

#%%
spectral=False
if spectral:
    model = SpectralCNN.from_CNN(model, fix_out = False).to(device)

#%% eval
data_sizing = ['TRIGO', 'BILINEAR']
model_sizing = ['NONE', 'TRIGO', 'BILINEAR']
combinations = [(d,m) for d in data_sizing for m in model_sizing]        
        
fname = 'results/FMNIST' + str(spectral * '-spectral') + '-3.csv'
size_step = 2
sizes = np.arange(3,60,size_step)
sizes = np.append(sizes, [28])
orig_size = [28,28]


model.eval()
accs = []
for d, m in combinations:
    accs_loc = []
    print(20*'<>')
    print('Starting test for data sizing: ' + d + ' and model sizing: ' + m)
    print(20*'<>')
    for s in sizes:
        acc = 0
        tot_steps = 0
        resize_data = sizing(d, [s,s])
        resize_model = sizing(m, orig_size)
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                # get batch data
                x, y = x.to(device), y.to(device)
                
                #resize input
                
                x = resize_data(x)
                
                
                # evaluate
                x = resize_model(x)
                pred = model(x)
                acc += (pred.max(1)[1] == y).sum().item()
                tot_steps += y.shape[0]
        print(20*'<>')
        print('Done for s='+str(s))
        print('Test Accuracy: ' + str(100 * acc/tot_steps) + str('[%]'))
        print(20*'<>')
        accs_loc.append(acc/tot_steps)
    accs.append([d,m] + accs_loc)
    
with open(fname, 'w') as f:
    writer = csv.writer(f, lineterminator = '\n')
    writer.writerow(['data sizing', 'model sizing'] + list(sizes))
    for i in range(len(accs)):
        writer.writerow(accs[i])
    
    

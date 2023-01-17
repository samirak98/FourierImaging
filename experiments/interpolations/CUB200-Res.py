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
from fourierimaging.modules import TrigonometricResize_2d
from fourierimaging.utils import datasets as data
import fourierimaging.train as train

#%% Set up variable and data for an example
experiment_file = '../classification/CUB200.yaml'
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

model = load_model(conf).to(device)
path = '../saved_models/resnet-18-CUB200'
model.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])

#%% eval
data_sizing = ['NONE','TRIGO', 'BILINEAR', 'NEAREST', 'BICUBIC']
model_sizing = ['NONE','TRIGO', 'BILINEAR', 'NEAREST', 'BICUBIC']
combinations = [(d,m) for d in data_sizing for m in model_sizing]

def select_sampling(name, size):
    resize = torch.nn.functional.interpolate
    if name == 'NONE':
        return lambda x: x
    if name == 'BILINEAR':
        return lambda x: resize(x, size=size, mode='bilinear')
        #return transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
    elif name == 'NEAREST':
        return transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST)
    elif name == 'BICUBIC':
        return lambda x: resize(x, size=size, mode='bicubic')
        #return transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
    elif name == 'TRILINEAR':
        return lambda x: resize(x, size=size, mode='trilinear')
    elif name == 'TRIGO':
        return TrigonometricResize_2d(size)
    else:
        raise ValueError('Unknown resize method: ' + name)
        
        
fname = 'results/CUB200-circular.csv'
size_step = 11
im_size = 224
sizes = np.arange(5,225,size_step)
orig_size = [im_size, im_size]


#%%
def main():
    model.eval()
    accs = []
    for d, m in combinations:
        accs_loc = []
        print(50*'.')
        print('Starting test for data sizing: ' + d + ' and model sizing: ' + m)
        print(50*'-')
        for s in sizes:
            acc = 0
            tot_steps = 0
            resize_data = select_sampling(d, [s,s])
            resize_model = select_sampling(m, orig_size)
            print(50*'.')
            print('Starting for s='+str(s))
            loader = valid_loader
            with torch.no_grad():
                for batch_idx, (x, y) in tqdm(enumerate(loader), total=len(loader)):
                    # get batch data
                    x, y = x.to(device), y.to(device)
                    #resize input 
                    x = resize_data(x)
                    # evaluate
                    x = resize_model(x)
                    pred = model(x)
                    acc += (pred.max(1)[1] == y).sum().item()
                    tot_steps += y.shape[0]
            print('Test accuracy [percentage]:', 100*acc/tot_steps)
            print(50*'-')
            accs_loc.append(acc/tot_steps)
        accs.append([d,m] + accs_loc)
        
    with open(fname, 'w') as f:
        writer = csv.writer(f, lineterminator = '\n')
        writer.writerow(['data sizing', 'model sizing'] + list(sizes))
        for i in range(len(accs)):
            writer.writerow(accs[i])
    
if __name__ == '__main__':
    main()
    

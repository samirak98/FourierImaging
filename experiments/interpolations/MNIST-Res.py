import torch
from torchvision import transforms
import yaml
import numpy as np
import csv

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
model = load_model(conf).to(device)
path = '../saved_models/perceptron-MNIST'
model.load_state_dict(torch.load(path, map_location=device))

#%% eval
data_sizing = ['TRIGO', 'BILINEAR', 'NEAREST', 'BICUBIC']
model_sizing = ['TRIGO', 'BILINEAR', 'NEAREST', 'BICUBIC']
combinations = [(d,m) for d in data_sizing for m in model_sizing]

def select_sampling(name, size):
    if name == 'BILINEAR':
        return transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
    elif name == 'NEAREST':
        return transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST)
    elif name == 'BICUBIC':
        return transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
    elif name == 'TRIGO':
        return TrigonometricResize_2d(size)
    else:
        raise ValueError('Unknown resize method: ' + name)
        
        
fname = 'results/MNIST.csv'
size_step = 1
sizes = np.arange(3,28+1,size_step)
orig_size = [28,28]


#%%
def main():
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
            resize_data = select_sampling(d, [s,s])
            resize_model = select_sampling(m, orig_size)
            
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
            print('Test Accuracy:', acc/tot_steps)
            print(20*'<>')
            accs_loc.append(acc/tot_steps)
        accs.append([d,m] + accs_loc)
        
    with open(fname, 'w') as f:
        writer = csv.writer(f, lineterminator = '\n')
        writer.writerow(['data sizing', 'model sizing'] + list(sizes))
        for i in range(len(accs)):
            writer.writerow(accs[i])
    
if __name__ == '__main__':
    main()
    

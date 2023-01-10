import torch
from torchvision import transforms
import yaml
import numpy as np
import csv
import matplotlib.pyplot as plt

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
    
#%% visulaize data
plt.close('all')
plt.style.use(['./teco.mplstyle'])
fig,ax = plt.subplots(1,4,figsize=(8.27,11.69/5))
accs = []
with open('results/MNIST-save.csv', 'r') as f:
    reader = csv.reader(f, lineterminator = '\n')
    old_data = None
    ax_idx=-1
    for i,row in enumerate(reader):
        if i == 0:
            sizes = row[2:]
        else:
            if old_data != row[0]:
                #ax = fig.add_subplot(ax_idx)#plt.figure()
                ax_idx+=1
                old_data = row[0]
                ax[ax_idx].set_title('Data sizing: ' + row[0])
                
            vals = np.array(row[2:], dtype=np.float64)
            ax[ax_idx].plot(sizes, vals, label=row[1])
            ax[ax_idx].legend()

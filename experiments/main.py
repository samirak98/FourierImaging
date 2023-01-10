import torch
import yaml

# custom imports
#%% custom imports
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

import fourierimaging as fi

from fourierimaging.utils.helpers import load_model, init_opt, fix_seed
from fourierimaging.utils import datasets as data
import fourierimaging.train as train

#%% Set up variable and data for an example
experiment_file = './classification/FMNIST.yaml'
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

#%% Initialize optimizer and lamda scheduler
opt, lr_scheduler = init_opt(model, conf['train']['opt'])
# initalize history
tracked = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
history = {key: [] for key in tracked}
trainer = train.trainer(model, opt, lr_scheduler, train_loader, valid_loader, conf['train'])


def main():
    print("Train model: " + conf['model']['type'])
    for i in range(conf['train']['epochs']):
        print(10*"<>")
        print(20*"|")
        print(10*"<>")
        print('Epoch', i)
        
        # train_step
        train_data = trainer.train_step()
        
        # validation step
        val_data = trainer.validation_step()
        
        # update history
        for key in tracked:
            if key in val_data:
                history[key].append(val_data[key])
            if key in train_data:
                history[key].append(train_data[key])
                
if __name__ == '__main__':
    main()
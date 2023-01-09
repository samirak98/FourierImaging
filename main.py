import torch
import yaml

# custom imports
from utils.helpers import load_model, init_opt, fix_seed
from utils import datasets as data
import train

#%% Set up variable and data for an example
experiment_file = 'utils/experiments/classification/CIFAR10.yaml'
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
opt = init_opt(model, conf['train']['opt'])
# initalize history
tracked = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
history = {key: [] for key in tracked}
trainer = train.trainer(model, opt, train_loader, valid_loader, conf['train'])


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
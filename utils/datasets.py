from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

import os


#%% 
def load(conf, test_size=1):
    train, valid, test, train_loader, valid_loader, test_loader = [None] * 6
    if conf['name'] == "MNIST":
        conf['im_shape'] = [1,28,28]
        
        # set mean and std for this dataset
        conf['mean'] = 0.1307
        conf['std'] = 0.3081

        # load MNIST
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST(conf['path'], train=True, download=conf['download'], transform=transform)
        test = datasets.MNIST(conf['path'], train=False, download=conf['download'], transform=transform)
    else:
        raise ValueError("Unknown dataset:" + conf.data_set)
   
    tr_loader, v_loader, te_loader = split_loader(train, test, conf['batch_size'],\
                                                  train_split=conf['train_split'],\
                                                  test_size=test_size,\
                                                  num_workers=conf['num_workers'])
        
    return tr_loader, v_loader, te_loader
#%% Define DataLoaders and split in train, valid, test       
def split_loader(train, test, batch_size, train_split=0.9, test_size=1,\
                 num_workers=1, seed=42):
    total_count = len(train)
    train_count = int(train_split * total_count)
    val_count = total_count - train_count
    generator=torch.Generator().manual_seed(seed)
    train, val = torch.utils.data.random_split(train,\
                                [train_count, val_count],generator=generator)

    if test_size != 1:
        test_count = int(len(test) * test_size)
        _count = len(test) - test_count
        test, _ = torch.utils.data.random_split(test, [test_count, _count])

    loader_kwargs = {'shuffle':True, 'pin_memory':True, 'num_workers':num_workers}
    train_loader = DataLoader(train, batch_size=batch_size, **loader_kwargs)
    valid_loader = DataLoader(val, batch_size=1000, **loader_kwargs)
    test_loader = DataLoader(test, batch_size=batch_size, **loader_kwargs)

    return train_loader, valid_loader, test_loader
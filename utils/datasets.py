from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

import os


#%% 
def load(conf):
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
    elif conf['name'] == "FashionMNIST":
        conf['im_shape'] = [1,28,28]
        
        # set mean and std for this dataset
        conf['mean'] = 0.5
        conf['std'] = 0.5

        # load MNIST
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.FashionMNIST(conf['path'], train=True, download=conf['download'], transform=transform)
        test = datasets.FashionMNIST(conf['path'], train=False, download=conf['download'], transform=transform)
    elif conf['name'] == "CIFAR10":
        conf['im_shape'] = [3,32,32]
        
        # set mean and std for this dataset
        conf['mean'] = torch.tensor([0.4914, 0.4822, 0.4465]).view(-1,1,1)
        conf['std'] = torch.tensor([0.2023, 0.1994, 0.2010]).view(-1,1,1)

        # load MNIST
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
        train = datasets.CIFAR10(conf['path'], train=True, download=conf['download'], transform=transform_train)
        test = datasets.CIFAR10(conf['path'], train=False, download=conf['download'], transform=transform_test)
    else:
        raise ValueError("Unknown dataset: " + conf['name'])
   
    tr_loader, v_loader, te_loader = split_loader(train, test,\
                                                  batch_size = conf['batch_size'],\
                                                  batch_size_test = conf['batch_size_test'],\
                                                  train_split=conf['train_split'],\
                                                  num_workers=conf['num_workers'])
        
    return tr_loader, v_loader, te_loader
#%% Define DataLoaders and split in train, valid, test       
def split_loader(train, test, batch_size=128, batch_size_test=100,\
                 train_split=0.9, num_workers=1, seed=42):
    total_count = len(train)
    train_count = int(train_split * total_count)
    val_count = total_count - train_count
    generator=torch.Generator().manual_seed(seed)
    train, val = torch.utils.data.random_split(train,\
                                [train_count, val_count],generator=generator)


    loader_kwargs = {'shuffle':True, 'pin_memory':True, 'num_workers':num_workers}
    train_loader = DataLoader(train, batch_size=batch_size, **loader_kwargs)
    valid_loader = DataLoader(val, batch_size=batch_size, **loader_kwargs)
    test_loader = DataLoader(test, batch_size=batch_size_test, **loader_kwargs)

    return train_loader, valid_loader, test_loader
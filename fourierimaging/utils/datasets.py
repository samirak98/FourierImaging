from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from omegaconf import DictConfig, OmegaConf, open_dict
from fourierimaging.modules import TrigonometricResize_2d
import torch
#import pandas as pd

import os


#%% 
def load(conf):
    valid = None
    if conf.name == "MNIST":
        conf.im_shape = [1,28,28]
        
        # set mean and std for this dataset
        conf.mean = 0.1307
        conf.std = 0.3081

        # load MNIST
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST(conf.path, train=True, download=conf.download, transform=transform)
        test = datasets.MNIST(conf.path, train=False, download=conf.download, transform=transform)
    elif conf.name == "FashionMNIST": 
        # load MNIST
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.FashionMNIST(conf.path, train=True, download=conf.download, transform=transform)
        test = datasets.FashionMNIST(conf.path, train=False, download=conf.download, transform=transform)
    elif conf.name == "CIFAR10":
        conf.im_shape = [3,32,32]
        
        # set mean and std for this dataset
        conf.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(-1,1,1)
        conf.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(-1,1,1)

        # load MNIST
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
        train = datasets.CIFAR10(conf.path, train=True, download=conf.download, transform=transform_train)
        test = datasets.CIFAR10(conf.path, train=False, download=conf.download, transform=transform_test)
    elif conf.name == "STANFORDCARS":
        im_shape = [3,224,224]
        conf.im_shape = [3,224,224]
        
        # set mean and std for this dataset
        conf.mean = 0.#torch.tensor([0.4914, 0.4822, 0.4465]).view(-1,1,1)
        conf.std = 1.#torch.tensor([0.2023, 0.1994, 0.2010]).view(-1,1,1)

        # transforms
        transform_train = transforms.Compose([
            transforms.Resize(tuple(im_shape[-2:])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(35),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.RandomPosterize(bits=2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
        ])
        transform_test = transforms.Compose([
            transforms.Resize(tuple(im_shape[-2:])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
        ])

        train = datasets.StanfordCars(conf.path, split='train', download=conf.download, transform=transform_train)
        test = datasets.StanfordCars(conf.path, split='test', download=conf.download, transform=transform_test)
        conf.num_classes = len(train.classes)
    elif conf.name == 'CUB200':
        im_shape = conf.im_shape
        transform = transforms.Compose([
                            #transforms.Resize(tuple(im_shape[-2:])),
                            transforms.ToTensor(),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.Resize(tuple(im_shape[-2:])),
                            #TrigonometricResize_2d(im_shape[-2:]),
                            ])

        transform_test = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize(tuple(im_shape[-2:])),
                            ])
        path = conf.path+'/CUB200'
        train = ImageFolder(path+'/train', transform=transform)
        with open_dict(conf):
            conf.num_classes = len(train.classes)
            conf.train_split=0.9
        test = ImageFolder(path+'/test', transform=transform_test)
        #valid = ImageFolder(path+'/valid', transform=transform)

        total_count = len(train)
        train_count = int(0.9*total_count)
        test_count = total_count - train_count
        # train, test = torch.utils.data.random_split(
        #                 train,
        #                 [train_count, test_count],
        #                 generator=torch.Generator().manual_seed(42)
        #             )


    else:
        raise ValueError("Unknown dataset: " + conf.name)
   
    tr_loader, v_loader, te_loader = split_loader(train, test,\
                                                  valid = valid,
                                                  batch_size = conf.batch_size,\
                                                  batch_size_test = conf.batch_size_test,\
                                                  train_split=conf.train_split,\
                                                  num_workers=conf.num_workers)
        
    return tr_loader, v_loader, te_loader
#%% Define DataLoaders and split in train, valid, test       
def split_loader(train, test, valid=None, batch_size=128, batch_size_test=100,\
                 train_split=0.9, num_workers=1, seed=42):
    total_count = len(train)
    train_count = int(train_split * total_count)
    val_count = total_count - train_count
    generator=torch.Generator().manual_seed(seed)

    loader_kwargs = {'pin_memory':True, 'num_workers':num_workers}
    if not (valid is None):
        valid_loader = DataLoader(valid, batch_size=batch_size, **loader_kwargs)
    elif val_count > 0:
        train, valid = torch.utils.data.random_split(train,\
                                    [train_count, val_count],generator=generator)
        valid_loader = DataLoader(valid, batch_size=batch_size, **loader_kwargs)
    else:
        valid_loader = None
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test, batch_size=batch_size_test, **loader_kwargs)
    return train_loader, valid_loader, test_loader
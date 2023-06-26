#%%
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torchvision import datasets

class myTransform:
    """Rotate by one of the given angles."""

    def __init__(self, kernel):
        self.conv2d = nn.Conv2d(1,1,kernel_size=2, padding='same', bias=False)
        self.conv2d.weight = torch.nn.Parameter(kernel)
        self.conv2d.weight.requires_grad_(False)
        self.relu = torch.nn.ReLU()
    def __call__(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        noise = torch.normal(mean = 0,std = 0.01,size=(1,x.shape[-2],x.shape[-1]))
        return x + noise

# %%

class CustomImageDataset(Dataset):
    def __init__(self,transform=None, target_transform=None, train = True):
        self.fmnist = datasets.FashionMNIST('../../datasets/', download=True, train = train,transform=transform)
        self.target_transform = target_transform

    def __len__(self):
        return self.fmnist.__len__()

    def __getitem__(self, idx):
        image,_ = self.fmnist.__getitem__(index=idx)
        label = torch.clone(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
# %%

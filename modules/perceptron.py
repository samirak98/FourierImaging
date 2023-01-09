import torch.nn as nn
import torch
import torch.nn.functional as F

class perceptron(nn.Module):
    def __init__(self, sizes, act_fun, mean = 0.0, std = 1.0):
        super(perceptron, self).__init__()
        
        self.act_fn = act_fun
        self.mean = mean
        self.std = std
        
        layer_list = [nn.Flatten()]
        for i in range(len(sizes)-1):
            layer_list.append(nn.Linear(sizes[i], sizes[i+1]))
            layer_list.append(self.act_fn())
            
        self.layers = nn.Sequential(*layer_list)
        
        
    def forward(self, x):
        x = (x - self.mean)/self.std
        return self.layers(x)


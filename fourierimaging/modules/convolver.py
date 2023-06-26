import torch.nn as nn
import torch

class Convolver(nn.Module):
    def __init__(self, act_fun=nn.ReLU()):
        super(Convolver, self).__init__()
        #
        self.act_fun = act_fun
        self.ksize1 = 2
        self.ksize2 = 2
        self.conv = torch.nn.Conv2d(1, 1, (self.ksize1,self.ksize2),\
                                    padding='same', bias = False)
        weight = torch.clone(self.conv.weight)
        weight[0,0,0] = 0
        self.conv.weight = torch.nn.Parameter(weight)
    def forward(self, x):
        x = self.conv(x)
        x = self.act_fun(x)

        return x

    def name(self):
        name = 'convolver'
        name += '-' + str(self.ksize1) + '-' + str(self.ksize2)
        return name
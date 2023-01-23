import torch.nn as nn
import torch

from .trigoInterpolation import SpectralConv2d, conv_to_spectral

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=1, stride=1, padding=0, padding_mode='zeros'):
        super(BasicBlock, self).__init__()
        self.kernel = kernel
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel, stride=stride,\
                                    padding = padding, padding_mode=padding_mode, bias = False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class SpectralBlock(nn.Module):
    def __init__(self, conv, im_shape, in_shape=None, out_shape=None, parametrization='spectral'):
        super(SpectralBlock, self).__init__()
        self.conv = conv_to_spectral(conv, im_shape,\
                                     parametrization=parametrization, norm='forward',\
                                     in_shape=in_shape, out_shape=out_shape)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class CNN(nn.Module):
    def __init__(self, mean = 0.0, std = 1.0, act_fun=nn.ReLU()):
        super(CNN, self).__init__()
        #
        self.mean = mean
        self.std = std
        self.act_fun = act_fun

        self.layers1 = BasicBlock(1,  64, kernel=5, stride=2, padding=2, padding_mode='circular')
        self.layers2 = BasicBlock(64, 64, kernel=5, stride=2, padding=2, padding_mode='circular')

        self.avgpool = nn.AdaptiveAvgPool2d((4,4))

        fc = [torch.nn.Linear(4 * 4 * 64, 128), self.act_fun, torch.nn.Linear(128, 10)]
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        x = (x - self.mean)/self.std
        x = self.layers1(x)
        x = self.layers2(x)

        # pool and classify
        x = self.avgpool(x)
        x = nn.Flatten()(x)
        x = self.fc(x)
        return x

class SpectralCNN(nn.Module):
    def __init__(self, CNN, fix_in = False, fix_out = False, parametrization='spectral'):
        super(SpectralCNN, self).__init__()
        self.mean = CNN.mean
        self.std = CNN.std
        self.act_fun = CNN.act_fun

        self.layers1 = SpectralBlock(CNN.layers1.conv, [28,28],\
                                     in_shape=self.select_shape([28, 28], fix_in),\
                                     out_shape=self.select_shape([28, 28], fix_out), parametrization=parametrization)
        self.layers2 = SpectralBlock(CNN.layers2.conv, [14,14], in_shape=self.select_shape([14, 14], fix_in), parametrization=parametrization)
        self.avgpool = CNN.avgpool # nn.AdaptiveAvgPool2d((4,4))

        self.fc = CNN.fc

    def forward(self, x):
        x = (x - self.mean)/self.std
        x = self.layers1(x)
        x = self.layers2(x)

        # pool and classify
        x = self.avgpool(x)
        x = nn.Flatten()(x)
        x = self.fc(x)
        return x

    def select_shape(self, im_shape, select):
        if select:
            return im_shape
        else:
            return None

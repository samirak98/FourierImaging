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
    def __init__(self, im_shape,\
                 in_channels=1, out_channels=1,\
                 in_shape=None, out_shape=None,\
                 parametrization='spectral',\
                 ksize1 = 1, ksize2 = 1,
                 stride = 1, norm = 'forward',\
                 odd = True):
        super(SpectralBlock, self).__init__()
        self.conv = SpectralConv2d(in_channels, out_channels,
                        out_shape=out_shape, in_shape=in_shape,\
                        parametrization = parametrization,\
                        ksize1 = ksize1, ksize2 = ksize2,\
                        stride = stride, norm = norm,\
                        odd = odd)

        self.relu = nn.ReLU(inplace=True)

    @classmethod
    def from_conv(cls, conv, im_shape,\
                  in_shape=None, out_shape=None,\
                  parametrization='spectral',\
                  norm='forward'):
        block = cls(im_shape)
        block.conv = conv_to_spectral(conv, im_shape,\
                                     parametrization=parametrization, norm=norm,\
                                     in_shape=in_shape, out_shape=out_shape)
        return block

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
    def __init__(self, mean = 0., std=1., act_fun = nn.ReLU(),\
                 fix_in = False, fix_out = False,\
                 parametrization='spectral'):
        super(SpectralCNN, self).__init__()
        self.mean = mean
        self.std = std
        self.act_fun = act_fun

        self.layers1 = SpectralBlock([28,28],\
                                    in_channels=1, out_channels=64,\
                                    ksize1 = 5, ksize2 = 5,\
                                    stride=(2,2),\
                                    in_shape=self.select_shape([28, 28], fix_in),\
                                    out_shape=self.select_shape([28, 28], fix_out),\
                                    parametrization=parametrization)
        self.layers2 = SpectralBlock([14,14],\
                                    in_channels=64, out_channels=64,
                                    ksize1 = 5, ksize2 = 5,\
                                    stride=(2,2),\
                                    in_shape=self.select_shape([14, 14], fix_in),\
                                    parametrization=parametrization)
        self.avgpool = nn.AdaptiveAvgPool2d((4,4))

        fc = [torch.nn.Linear(4 * 4 * 64, 128), self.act_fun, torch.nn.Linear(128, 10)]
        self.fc = nn.Sequential(*fc)

    @classmethod
    def from_CNN(cls, CNN, fix_in = False, fix_out = False, parametrization='spectral'):
        model = cls(mean=CNN.mean, std = CNN.std,\
                    act_fun = CNN.act_fun,\
                    fix_in = fix_in, fix_out = fix_out)
        model.layers1 = SpectralBlock.from_conv(CNN.layers1.conv, [28,28],\
                                    in_shape=model.select_shape([28, 28], fix_in),\
                                    out_shape=model.select_shape([28, 28], fix_out),\
                                    parametrization=parametrization)
        model.layers2 = SpectralBlock.from_conv(CNN.layers2.conv, [14,14],\
                                    in_shape=model.select_shape([14, 14], fix_in),\
                                    parametrization=parametrization)
        model.avgpool = CNN.avgpool
        model.fc = CNN.fc

        return model

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

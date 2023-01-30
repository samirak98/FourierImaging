import torch.nn as nn
import torch

from .trigoInterpolation import SpectralConv2d, conv_to_spectral

class SpectralBlock(nn.Module):
    def __init__(self,
                 in_channels=1, out_channels=1,
                 in_shape=None, out_shape=None,
                 parametrization='spectral',
                 ksize1 = 1, ksize2 = 1,
                 stride = 1, norm = 'forward',
                 odd = True,
                 conv_like_cnn = False):
        super(SpectralBlock, self).__init__()
        self.conv = SpectralConv2d(in_channels, out_channels,
                        out_shape=out_shape, in_shape=in_shape,\
                        parametrization = parametrization,\
                        ksize1 = ksize1, ksize2 = ksize2,\
                        stride = stride, norm = norm,\
                        odd = odd,
                        conv_like_cnn = conv_like_cnn)

        self.relu = nn.ReLU(inplace=True)

    @classmethod
    def from_conv(cls, conv, im_shape,
                  in_shape=None, out_shape=None,
                  parametrization='spectral',
                  norm='forward',
                  conv_like_cnn = False,
                  ksize=None):
        block = cls()
        block.conv = conv_to_spectral(
                        conv, im_shape,
                        parametrization=parametrization, norm=norm,\
                        in_shape=in_shape, out_shape=out_shape,
                        conv_like_cnn = conv_like_cnn,
                        ksize=ksize
                    )
        return block

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class SpectralCNN(nn.Module):
    def __init__(self, mean = 0., std=1., act_fun = nn.ReLU(),\
                 fix_in = False, fix_out = False,\
                 mid_channels=32, out_channels=64,\
                 ksize1 = 5, ksize2 = 5,\
                 parametrization='spectral',
                 norm='forward',
                 conv_like_cnn = False,
                 im_shape = [28,28],
                 stride = 2):
        super(SpectralCNN, self).__init__()
        self.mean = mean
        self.std = std
        self.act_fun = act_fun
        self.ksize1 = ksize1
        self.ksize2 = ksize2
        self.parametrization = parametrization
        self.norm = norm
        self.stride = stride

        self.layers1 = SpectralBlock(
                            in_channels=1, out_channels=mid_channels,\
                            ksize1 = ksize1, ksize2 = ksize2,\
                            stride = (stride, stride),\
                            in_shape = self.select_shape([28, 28], fix_in),\
                            out_shape = self.select_shape([28, 28], fix_out),\
                            parametrization = parametrization,
                            norm = norm,
                            conv_like_cnn = conv_like_cnn,
                        )

        second_imshape = [im_shape[0]//stride, im_shape[1]//stride]
        self.layers2 = SpectralBlock(
                            in_channels=mid_channels, out_channels=out_channels,
                            ksize1 = min(ksize1, second_imshape[0]), ksize2 = min(ksize2, second_imshape[1]),
                            stride=(stride, stride),
                            in_shape=self.select_shape([14, 14], fix_in),
                            parametrization=parametrization,
                            norm = norm,
                            conv_like_cnn = conv_like_cnn,
                        )
        self.avgpool = nn.AdaptiveAvgPool2d((4,4))

        fc = [torch.nn.Linear(4 * 4 * out_channels, 128), self.act_fun, torch.nn.Linear(128, 10)]
        self.fc = nn.Sequential(*fc)

    @classmethod
    def from_CNN(
                cls, CNN, fix_in = False, fix_out = False,
                parametrization='spectral', norm='forward',
                conv_like_cnn = False,
                im_shape = [28,28],
                ksize=None):

        model = cls(mean=CNN.mean, std = CNN.std,\
                    act_fun = CNN.act_fun,\
                    fix_in = fix_in, fix_out = fix_out,
                    norm=norm,
                    ksize1 = CNN.ksize1,
                    ksize2 = CNN.ksize2,
                    conv_like_cnn = conv_like_cnn,
                    parametrization=parametrization
                    )
        model.layers1 = SpectralBlock.from_conv(
                            CNN.layers1.conv, im_shape,\
                            in_shape=model.select_shape([28, 28], fix_in),\
                            out_shape=model.select_shape([28, 28], fix_out),\
                            parametrization=parametrization,
                            norm=norm,
                            conv_like_cnn = True,
                            ksize=ksize
                        )

        second_imshape = [im_shape[0]//CNN.stride, im_shape[1]//CNN.stride]
        model.layers2 = SpectralBlock.from_conv(
                            CNN.layers2.conv, second_imshape,
                            in_shape=model.select_shape([14, 14], fix_in),
                            parametrization=parametrization,
                            norm=norm,
                            conv_like_cnn = True,
                            ksize=ksize
                        )
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

    def name(self):
        name = 'spectral-cnn'
        name += '-' + self.parametrization
        name += '-' + str(self.ksize1) + '-' + str(self.ksize2)
        return name
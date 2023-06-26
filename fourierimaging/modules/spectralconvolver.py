import torch.nn as nn
import torch

from .trigoInterpolation import SpectralConv2d, conv_to_spectral
class SpectralConvolver(nn.Module):
    def __init__(self, act_fun = nn.ReLU(),\
                 ksize1=1, ksize2=1,\
                 fix_in = False, fix_out = False,\
                 norm='forward',
                 im_shape = [28,28]):
        super(SpectralConvolver, self).__init__()
        self.act_fun = act_fun
        self.ksize1 = ksize1
        self.ksize2 = ksize2
        self.parametrization = 'spectral'
        self.norm = norm

    @classmethod
    def from_Convolver(
                cls, Convolver, fix_in = False, fix_out = False,
                parametrization='spectral', norm='forward',
                conv_like_cnn = False,
                im_shape = [28,28]):

        model = cls(act_fun = Convolver.act_fun,\
                    fix_in = fix_in, fix_out = fix_out,
                    norm=norm,
                    ksize1 = Convolver.ksize1,
                    ksize2 = Convolver.ksize2
                    )
        model.spec_conv = conv_to_spectral(Convolver.conv, im_shape,
                        parametrization=parametrization, norm=norm,\
                        in_shape=None, out_shape=None,
                        conv_like_cnn = True,
                        ksize=(28,28)
                    )

        return model

    def forward(self, x):
        x = self.spec_conv(x)
        x = self.act_fun(x)

        return x

    def name(self):
        name = 'spectral-cnn'
        name += '-' + self.parametrization
        name += '-' + str(self.ksize1) + '-' + str(self.ksize2)
        return name
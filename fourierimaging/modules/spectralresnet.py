from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.models._api import Weights, WeightsEnum
from .trigoInterpolation import SpectralConv2d, conv_to_spectral

def conv1x1(in_planes: int, out_planes: int, stride: int = 1, padding_mode='zeros') -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding_mode=padding_mode)

class SpectralBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        in_shape=None, out_shape=None,\
        parametrization='spectral',\
        ksize1 = 1, ksize2 = 1,
        norm = 'forward',\
        conv_like_cnn: bool = False,
        odd = True
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = SpectralConv2d(
                        inplanes, planes,
                        out_shape=out_shape, in_shape=in_shape,
                        parametrization = parametrization,
                        ksize1 = ksize1, ksize2 = ksize2,
                        stride = (stride,stride), norm = norm,
                        odd = odd,
                        conv_like_cnn = conv_like_cnn
                    )

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SpectralConv2d(
                        planes, planes,
                        out_shape=out_shape, in_shape=in_shape,
                        parametrization = parametrization,
                        ksize1 = ksize1, ksize2 = ksize2,
                        stride = (1,1), norm = norm,
                        odd = odd,
                        conv_like_cnn = conv_like_cnn
                    )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SpectralResNet(nn.Module):
    def __init__(
        self,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        ksize1: int = 5, ksize2: int = 5,
        conv_like_cnn: bool = False,
        norm: str = 'forward',
        parametrization: str = 'spectral',
        fix_in: bool = False,
        fix_out: bool = False,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.parametrization = parametrization
        self.norm = norm
        self.ksize1=ksize1
        self.ksize2=ksize2
        self.fix_in = fix_in
        self.fix_out = fix_out
        self.conv_like_cnn = conv_like_cnn

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = SpectralConv2d(
                        3, self.inplanes,
                        in_shape = self.select_shape([28, 28], fix_in),
                        out_shape = self.select_shape([28, 28], fix_out),
                        parametrization = self.parametrization,
                        ksize1 = 7, ksize2 = 7,
                        stride = (2,2), 
                        norm = self.norm,
                        conv_like_cnn = conv_like_cnn
                    )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    @classmethod
    def from_resnet(cls, im_shape, **kwargs):
        model = cls(**kwargs)

        self.conv1 = conv_to_spectral(
                            model.conv1, im_shape,\
                            in_shape=model.select_shape(im_shape, fix_in),\
                            out_shape=model.select_shape(im_shape, fix_out),\
                            parametrization=model.parametrization,
                            norm=model.norm,
                            conv_like_cnn = model.conv_like_cnn
                        )
        self.layer1 = model._convert_layer(model.layer1, [im_shape[0]//2, im_shape[1]//2])
        self.layer2 = model._convert_layer(model.layer2, [im_shape[0]//4, im_shape[1]//4])
        self.layer3 = model._convert_layer(model.layer1, [im_shape[0]//8, im_shape[1]//8])
        self.layer4 = model._convert_layer(model.layer1, [im_shape[0]//16, im_shape[1]//16])

    def _convert_layer(self, layer, im_shape):
        for m in layers:
            l =[]
            if isinstance(m, nn.Conv2d):
                print(m)
            else:
                l.append(m)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(
            SpectralBlock(
                inplanes=self.inplanes, planes=planes,
                ksize1 = self.ksize1, ksize2 = self.ksize2,
                stride = stride,
                downsample = downsample,
                in_shape = self.select_shape([28, 28], self.fix_in),
                out_shape = self.select_shape([28, 28], self.fix_out),
                parametrization = self.parametrization,
                norm = self.norm,
                conv_like_cnn = self.conv_like_cnn
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                SpectralBlock(
                    inplanes=self.inplanes, planes=planes,
                    ksize1 = self.ksize1, ksize2 = self.ksize2,
                    stride = 1,
                    in_shape = self.select_shape([28, 28], self.fix_in),
                    out_shape = self.select_shape([28, 28], self.fix_out),
                    parametrization = self.parametrization,
                    norm = self.norm,
                    conv_like_cnn = self.conv_like_cnn
                )
            )

        return nn.Sequential(*layers)

    def select_shape(self, im_shape, select):
        if select:
            return im_shape
        else:
            return None

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def name(self,):
        return 'spectral-resnet'


def _resnet(
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> SpectralResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = SpectralResNet(layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def spectralresnet18(*, weights = None, progress: bool = True, **kwargs: Any) -> SpectralResNet:
    #weights = ResNet18_Weights.verify(weights)
    return _resnet([2, 2, 2, 2], weights, progress, **kwargs)
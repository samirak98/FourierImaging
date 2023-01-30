from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np

from torchvision.models._api import Weights, WeightsEnum
from .trigoInterpolation import SpectralConv2d, conv_to_spectral, TrigonometricResize_2d
from .resnet import BasicBlock

class conv_trigo_stride(nn.Module):
    def __init__(self,
                in_planes: int, 
                out_planes: int, 
                stride: int = 1, 
                kernel_size: int = 1,
                groups: int = 1, 
                dilation: int = 1,
                padding:int = 0,
                bias:bool = False,
                padding_mode='zeros'):

        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(
                        in_planes,
                        out_planes,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        groups=groups,
                        bias=bias,
                        dilation=dilation,
                        padding_mode=padding_mode,
                    )
        self.resize = TrigonometricResize_2d

    def forward(self, x):
        x = self.conv(x)
        stride_size = [int(np.ceil(x.shape[-2]/self.stride)), int(np.ceil(x.shape[-1]/self.stride))] 
        x = self.resize([stride_size[0], stride_size[1]])(x)
        return x


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, padding_mode='zeros',
            stride_trigo: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    if not stride_trigo:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding_mode=padding_mode)
    else:
        return conv_trigo_stride(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
            padding_mode=padding_mode,
        )

class SpectralBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int = 1,
        planes: int = 1,
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
        odd = True,
        stride_trigo = False
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
                        conv_like_cnn = conv_like_cnn,
                        stride_trigo = stride_trigo
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
                        conv_like_cnn = conv_like_cnn,
                        stride_trigo = stride_trigo
                    )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    @classmethod
    def from_resblock(
                    cls, resblock, im_shape,
                    in_shape=None, out_shape=None,
                    parametrization='spectral',
                    norm='forward',
                    conv_like_cnn = False,
                    stride_trigo = False
                    ):
        block = cls()
        block.stride = resblock.stride
        block.bn1 = resblock.bn1
        block.relu = resblock.relu 

        if stride_trigo:
            resconv1 = resblock.conv1.conv
            resconv2 = resblock.conv2.conv
            stride   = resblock.stride
        else:
            stride = None
            resconv1 = resblock.conv1
            resconv2 = resblock.conv2

        block.conv1 = conv_to_spectral(
                        resconv1, im_shape,
                        parametrization=parametrization, norm=norm,\
                        in_shape=in_shape, out_shape=out_shape,
                        conv_like_cnn = True,
                        stride_trigo = stride_trigo,
                        stride = (stride, stride)
                    )
        
        s_shape = [np.ceil(im_shape[0]/block.stride), np.ceil(im_shape[1]/block.stride)]
        block.conv2 = conv_to_spectral(
                        resconv2, np.array(s_shape, dtype=int),
                        parametrization=parametrization, norm=norm,\
                        in_shape=in_shape, out_shape=out_shape,
                        conv_like_cnn = True,
                        stride_trigo = stride_trigo
                    )

        block.bn2 = resblock.bn2
        block.downsample = resblock.downsample
        #print(block.conv1)
        return block


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
        stride_trigo: bool = False
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
        self.stride_trigo = stride_trigo

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
                        conv_like_cnn = conv_like_cnn,
                        stride_trigo = stride_trigo
                    )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        #self.resize = lambda x: nn.functional.interpolate(x, size=[4,4], mode='bicubic', align_corners=False, antialias=True)
        self.resize = TrigonometricResize_2d([4,4])
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
    def from_resnet(cls, resnet, im_shape,\
                    fix_in = False,
                    fix_out= False,
                    norm='forward',
                    stride_trigo = False,
                    ):
        layers = [
                len(resnet.layer1), 
                len(resnet.layer2), 
                len(resnet.layer3), 
                len(resnet.layer4)
                ]

        model = cls(layers,
                    fix_in = False,
                    fix_out= False,
                    norm=norm,
                    stride_trigo = stride_trigo
                    )

        if stride_trigo:
            resconv = resnet.conv1.conv
        else:
            resconv = resnset.conv1
        model.conv1 = conv_to_spectral(
                            resconv, im_shape,\
                            in_shape=model.select_shape(im_shape, fix_in),\
                            out_shape=model.select_shape(im_shape, fix_out),\
                            parametrization=model.parametrization,
                            norm=norm,
                            conv_like_cnn = True,
                            stride_trigo = stride_trigo,
                            stride = (2,2)
                        )
        model.bn1 = resnet.bn1
        model.maxpool = resnet.maxpool



        current_shape = [im_shape[0]//4, im_shape[1]//4]
        for i in range(1,5):
            l_name = 'layer' + str(i)
            reslayer = getattr(resnet, l_name)

            setattr(model, 
                    l_name, 
                    model._convert_layer(
                        reslayer,
                        current_shape,
                        norm=norm
                        )
                    )

            stride = reslayer[0].stride
            current_shape = [int(np.ceil(current_shape[0]/stride)), int(np.ceil(current_shape[1]/stride))]

        # model.layer2 = model._convert_layer(resnet.layer2, [im_shape[0]//4, im_shape[1]//4])
        # model.layer3 = model._convert_layer(resnet.layer3, [im_shape[0]//8, im_shape[1]//8])
        # model.layer4 = model._convert_layer(resnet.layer4, [im_shape[0]//16, im_shape[1]//16])

        model.avgpool = resnet.avgpool
        model.fc = resnet.fc
        return model

    def _convert_layer(self, layer, im_shape, norm='forward'):
        l = []
        current_shape = im_shape
        for m in layer:           
            if isinstance(m, BasicBlock):
                mm = SpectralBlock.from_resblock(m, current_shape, norm=norm, stride_trigo=self.stride_trigo)
                current_shape = [int(np.ceil(current_shape[0]/m.stride)),  int(np.ceil(current_shape[1]/m.stride))]
                
                l.append(mm)
            else:
                l.append(m)

        return nn.Sequential(*l)

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
                conv1x1(self.inplanes, planes, stride, stride_trigo=self.stride_trigo),
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

        x = self.resize(x)

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
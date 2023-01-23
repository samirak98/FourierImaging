import torch
from torchvision import transforms
from fourierimaging.modules import TrigonometricResize_2d

def sizing(name, size):
    resize = torch.nn.functional.interpolate
    if name == 'BILINEAR':
        return lambda x: resize(x, size=size, mode='bilinear', align_corners=False, antialias=True)
    elif name == 'NONE':
        return lambda x:x
    elif name == 'NEAREST':
        return transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST)
    elif name == 'BICUBIC':
        return lambda x: resize(x, size=size, mode='bicubic', align_corners=False, antialias=True)
        #return transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
    elif name == 'TRILINEAR':
        return lambda x: resize(x, size=size, mode='trilinear', align_corners=False, antialias=True)
    elif name == 'TRIGO':
        return TrigonometricResize_2d(size)
    else:
        raise ValueError('Unknown resize method: ' + name)
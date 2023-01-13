import torch

x = torch.rand((1,1,9,9))

resize = torch.nn.functional.interpolate
xx = resize(x, size=(3,3), mode='bilinear', align_corners=False, antialias=False)

print(x)
print(xx)
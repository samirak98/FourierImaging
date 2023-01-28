import torch
import torch.fft as fft
import torch.nn as nn

#%%
x = torch.rand((1,1,28,28))
w = torch.rand((1,1,28,28))

#%%
class C:
    def __init__(self, w, norm='forward'):
        self.w = w.clone()
        self.w = nn.Parameter(w)
        self.norm = norm

    def __call__(self, x):
        g = fft.fft2(self.w, norm=self.norm) * fft.fft2(x, norm=self.norm)
        g = fft.ifft2(g, norm=self.norm)

        z = torch.sum(g, axis=[-2,-1])
        return z

g = []
norms = ['forward', 'backward']
for norm in norms:
    F = C(w, norm=norm)
    l = F(x)
    l.backward()
    g.append(F.w.grad)

print(torch.norm(g[0]-g[1]))
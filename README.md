# Discretization-invariant Image Processing based on Fourier Neural Operators

Link to the mind map https://coggle.it/diagram/Y7rvDr1Eo51j0Lsw/t/ssvm-fno

## SpectralConv2d

- ```in_channels, out_channels```: clear
- ```paramterization```: 'spatial', 'spectral' how to parameterize the kernel. Convertable, but differences in training opt
- ```in_shape```: only applies for ```param='spatial'```. Options:
    - ```in_shape=None```: spatial kernel doman varies with input resolution in the same way as normal CNNS
    - ```in_shape=[N,M]```: kernel to image relation is apadpted to kernel shape to ```in_shape``` relation. 
- ```out_shape```: applies for both parametrizations Determines output shape after convolution but **before** striding.
    - ```out_shape=None```: this yields ```out_shape``` equal to the shape of the input
    - ```out_shape=[N,M]```: resizng via trigonometric interpolation. Here, ```N,M``` can not depend on the input shape.
- ```stride```: usual striding, like for CNNs.
- ```odd```: only applies for ```param='spectral'```. Determines if the parametrized kernel fft $\mathcal{F}(k)$ has odd or even width.
- ```norm```: determines the norm to be used in FFT. If ```out_shape``` is **not** ```None``` the norm is recommended to be choosen as ```norm='forward'```. 

> A normal CNN can be mimicked by using ```in_shape=None, out_shape = None```
>


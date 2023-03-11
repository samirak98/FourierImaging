import numpy as np
import torch
import torch.fft as fft
import torchvision.transforms.functional as TF
import torch.nn.functional as tf
import torch.nn as nn

def rfftshift(x):
    """Returns a shifted version of a matrix obtained by torch.fft.rfft2(x), i.e. the left half of torch.fft.fftshift(torch.fft.fft2(x))"""
    # Real FT shifting thus consists of two steps: 1d-Shifting the last dimension and flipping in second to last dimension and taking the complex conjugate,
    # to get the frequency order [-N,...,-1,0]
    return torch.conj(torch.flip(fft.fftshift(x, dim=-2), dims=[-1]))

def irfftshift(x):
    """Inverts rfftshift: Returns a non-shifted version of the left half of torch.fft.fftshift(torch.fft.fft2(x)), i.e. torch.fft.rfft2(x)"""
    return torch.conj(torch.flip(fft.ifftshift(x, dim=-2), dims=[-1]))

def symmetric_padding(xf_old, im_shape_old, im_shape_new):
    """Returns the rfft2 of the real trigonometric interpolation of an image x of shape 'im_shape_old' and shifted real Fourier transform 'xf_old' to shape 'im_shape_new'"""
    #xf_old has to be a shifted rfft2 matrix
    add_shape = np.array(xf_old.shape[:-2]) #[batchsize, channels]

    ft_height_old = im_shape_old[-2] + (1 - im_shape_old[-2]%2) #always odd
    ft_width_old = im_shape_old[-1]//2 + 1 #always odd
    ft_height_new = im_shape_new[-2] + (1 - im_shape_new[-2]%2) #always odd
    ft_width_new  = im_shape_new[-1]//2 + 1 #always odd
    xf_shape = tuple(add_shape)  + (ft_height_old, ft_width_old) # shape for the unpadded trafo array

    pad_height = (ft_height_new - ft_height_old)//2 #the difference between both ft shapes is always even
    pad_width = (ft_width_new - ft_width_old)
    pad_list = [pad_width, 0, pad_height, pad_height] #'starting from the last dimension and moving forward, (padding_left,padding_right, padding_top,padding_bottom)'
    xf = torch.zeros(size=xf_shape, dtype=torch.cfloat, device=xf_old.device)

    xf[..., :im_shape_old[-2], :] = xf_old #for odd dimensions, this already represents a set of Herm.sym. coefficients

    # for even dimensions, the coefficients corresponding to the nyquist frequency are split symmetrically
    if im_shape_old[-2]%2 == 0:
        xf[...,  0, :] *= 0.5 # nyquist_row /2
        xf[..., -1, :] = xf[..., 0, :] # nyquist_row/2
        
    if im_shape_old[-1]%2 == 0:
        xf[..., :,  0] *= 0.5 # nyquist_col/2
    
    ## Trigonometric interpolation: Zero-Padding/Cut-Off of the symmetric (odd-dimensional) Fourier transform, if needed convert to even dimensions
    #Zero-padding/cut-off to odd_dimension such that desired_dimension <= odd_dimension <= desired_dimension + 1
    xf_pad = tf.pad(xf, pad_list)

    #if desired dimension is even, reverse the nyquist splitting
    if im_shape_new[-2]%2 == 0:
        xf_pad[..., 0, :] *= 2
                
    if im_shape_new[-1]%2 == 0:
        xf_pad[..., :, 0] *= 2

    return xf_pad[..., :im_shape_new[-2], :]
    

class TrigonometricResize_2d:
    """Resize 2d image with trigonometric interpolation"""

    def __init__(self, shape, norm = 'forward', check_comp=False):
        self.shape = shape
        self.norm = norm
        self.check_comp = check_comp
    
    def __call__(self, x):
        im_shape_new = np.array(self.shape)

        if torch.is_complex(x): # for complex valued functions, trigonometric interpolations is done by simple zero-padding of the Fourier coefficients
            x_inter = fft.ifft2(fft.fft2(x, norm=self.norm), s=self.shape, norm=self.norm)
        else: # for real valued functions, the coefficients have to be Hermitian symmetric
            im_shape_old = np.array(x.shape[-2:]) #this has to be saved since it cannot be uniquely obtained by rfft(x)
            x_inter = fft.irfft2(irfftshift(symmetric_padding(rfftshift(fft.rfft2(x, norm = self.norm)), im_shape_old, im_shape_new)), s=tuple(im_shape_new), norm=self.norm)
        return x_inter

    def check_symmetry(self, x, im_shape=None, threshold=1e-5):
        ## Helper function to check Hermitian symmetry of an odd-dimensioned matrix of Fourier coefficients. 
        # Symmetry is fulfilled iff the coefficients correspond to a function which attains only real values at the considered sampling points
        if self.check_comp:
            x_flip = torch.flip(x, dims=[-2,-1])
            symmetry_check = x - torch.conj(x_flip)
            symmetry_norm = torch.norm(symmetry_check, p=float("Inf"))
            if  symmetry_norm > threshold:
                print('Not symmetric: (norm: ' + str(symmetry_norm) + ' for old shape: ' + str(im_shape))

    def check_imag(self, x, im_shape=None, threshold=1e-5):
        if self.check_comp:
            imag_norm = torch.norm(x.imag, p=float("Inf"))
            if  imag_norm > threshold:
                print('The imaginary part of the image is unusual high, norm: ' +str(imag_norm)\
                    + ' for old shape: ' + str(im_shape))


# This is a modified version of https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
# Modifications and additions are as follows:
# - additional parameter 'weight' = None or tensor of in_channels x out_channels x ksize1 x ksize2: if not None, the spectral weights are set according to the passed argument
# - additional parameter 'out_shape' = [int, int]: determines height and width of output. If not chosen, output shape will be the same as input shape. If fixed and norm = 'forward', FNO-behavior is equivalent to CNN-behavior for trigonometrically interpolated inputs.
# - additional parameter 'in_shape' = [int, int]: determines the image dimension that corresponds to the parameters.
# - additional parameter 'parameterization' = {'spectral', 'spatial'}: determines wether the optimization is done in spectral (original) or spatial (addition) domain
# - additional parameters 'ksize1' and 'ksize2': determines kernel height and width and substitutes 'modes1' and 'modes2'. This makes it possible to use even-dimensional parameters. Only necessary if 'weight' is None.
# - additional parameters 'stride' = [int,int], 'strided_trigo' = {True, False}: determines size of stride and if the dimensionality reduction is done by conventional striding (strided_trigo = False) or downsizing with trigonometric interpolation (strided_trigo = True)
# - additional parameter 'norm' = {'forward', 'backward', 'ortho'}: Normalization factor used for fft-functions. If out_shape is fixed, it is recommended to use 'forward' to be interpolation-equivariant w.r.t. trigonometric interpolation
# - additional parameter 'odd' = {True, False}: Specifies the oddity of the width of the spectral kernel, since this is not clear from the rfft-representation.
# - additional parameter 'conv_like_cnn': Should be True if parameters are taken from a standard CNN-layer with even-dimensional kernel or even-dimensional (training) inputs to resemble CNN-behavior
# - changes in parametrization: spectral parameters are now stored in one tensor (corresponds to a shifted version of the split parametrization in original code)
# - functional changes: behavior for even dimensions is now in accordance to trigonometric interpolation of real-valued functions
class SpectralConv2d(nn.Module):
    """2D spectral convolution layer
    
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        weight (tensor): spectral weights of shape (in_channels, out_channels, ksize1, ksize2)
        out_shape (list): determines height and width of output. If not chosen, output shape will be the same as input shape
        in_shape (list): determines the image dimension that corresponds to the parameters
        parametrization (str): determines wether the optimization is done in spectral or spatial domain
        ksize1 (int): determines kernel height
        ksize2 (int): determines kernel width
        stride (list): determines size of stride
        strided_trigo (bool): determines if the dimensionality reduction is done by conventional striding or downsizing with trigonometric interpolation
        norm (str): Normalization factor used for fft-functions
        odd (bool): Specifies the oddity of the width of the spectral kernel, since this is not clear from the rfft-representation
        conv_like_cnn (bool): If True resembles CNN-behavior for a certain resolution
        """

    def __init__(self, in_channels, out_channels, weight=None,
                 out_shape=None, in_shape=None, parametrization = 'spectral',
                 ksize1 = 1, ksize2 = 1,
                 stride = (1,1), stride_trigo = False,
                 norm = 'forward', odd = True, conv_like_cnn = False
                ):
        super(SpectralConv2d, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = out_shape 
        self.in_shape = in_shape
        self.parametrization = parametrization
        self.stride = stride
        self.norm = norm
        self.odd = odd
        self.conv_like_cnn = conv_like_cnn 
        self.stride_trigo = stride_trigo
        im_factor = 1
        
    
        #Initialize weight according to chosen parametrization
        if parametrization == 'spectral':
            if not in_shape is None:
                if norm == 'forward':
                    im_factor = np.prod(in_shape)
                elif norm == 'ortho':
                    im_factor = np.sqrt(np.prod(in_shape))
                
            if weight is None:   
                self.scale = 1 / (in_channels * out_channels)

                weight = torch.rand(in_channels, out_channels,
                                    ksize1, ksize2//2+1,
                                    dtype=torch.cfloat
                                    )
                weight = self.scale * (weight)
                        
                self.odd = ksize2%2
                weight[:,:,0,0].imag = 0.  
            self.ksize1 = weight.shape[-2]
            #self.ksize2 = weight.shape[-1]*2 - (2 - self.odd)  
            self.ksize2 = 2 * (weight.shape[-1] - 1) + self.odd               
        elif parametrization == 'spatial':
            if in_shape is None:
                im_factor = 1
            else:
                #if norm == 'forward':
                #im_factor = np.prod(in_shape)
                im_factor=1

            self.scale = 1 / (in_channels * ksize1 * ksize2)

            if weight is None:        
                weight = im_factor * np.sqrt(self.scale) * 2 * (torch.rand(in_channels, out_channels, ksize1, ksize2) - 0.5)
            self.ksize1 = weight.shape[-2]
            self.ksize2 = weight.shape[-1]
        self.weight = nn.Parameter(weight.clone())
        
    # Complex multiplication (original)
    def compl_mul2d(self, input, weight):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weight)

    def forward(self, x):
        x_shape = np.array(x.shape[-2:])

        if self.in_shape == None:
            im_shape_old = x_shape # variable spatial support of kernel in case of spatial parametrization
        else:
            im_shape_old = np.array(self.in_shape) # 'fixed' spatial support in case of spatial and spectral parametrization

        if self.out_shape == None:
            im_shape_new = x_shape
        else:
            im_shape_new = np.array(self.out_shape)

        # adapt the weight parameters to input dimension by trigonometric interpolation (to odd dimension)
        if self.parametrization=='spectral':
            kernel_shape = np.array([self.ksize1, self.ksize2])
            multiplier_padded = symmetric_padding(self.weight, kernel_shape, im_shape_new + (1-im_shape_new%2))
        elif self.parametrization=='spatial':
            multiplier = spatial_to_spectral(self.weight, im_shape_old, norm=self.norm, conv_like_cnn=self.conv_like_cnn)
            # spatial zero-padding to match image shape, then ifftshift to align center, then rfft2 and rfftshift to get multiplier, maybe this can be optimized by using the 's' parameter for rfft2
            multiplier_padded = symmetric_padding(multiplier,\
                                                  im_shape_old,\
                                                  im_shape_new + (1-im_shape_new%2))
            

        #For even input dimensions we interpolate to the next higher odd dimension
        #this could be optimized by checking dimensions first
        x_ft_padded =   symmetric_padding(
                            rfftshift(
                                fft.rfft2(x, norm = self.norm)
                            ), 
                            x_shape, im_shape_new + (1-im_shape_new%2)
                        )

        # Elementwise multiplication of rfft-coefficients and return to physical space after correcting dimension with symmetric padding if desired dimension is even
        output = fft.irfft2(
                    irfftshift(
                        symmetric_padding(
                            self.compl_mul2d(x_ft_padded, multiplier_padded),\
                            im_shape_new + (1-im_shape_new%2), im_shape_new
                        )
                    ),
                    norm = self.norm,\
                    s=tuple(im_shape_new)
                )
        # dimension reduction by trigonometric interpolation or conventional striding (not resolution invariant)
        if sum(self.stride) > 1:
            if self.stride_trigo:
                stride_size = [int(np.ceil(im_shape_new[0]/self.stride[0])), int(np.ceil(im_shape_new[1]/self.stride[1]))] 
                output = TrigonometricResize_2d([stride_size[0], stride_size[1]])(output)
            else:
                output = output[...,0::self.stride[0], 0::self.stride[1]]

        return output


    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={ksize1}'
             ', stride={stride}')
        return s.format(**self.__dict__)

def spatial_to_spectral(weight, im_shape, 
                        norm='forward', conv_like_cnn=False,
                        ksize=None):
    weight = weight.clone()
    kernel_shape = np.array([weight.shape[-2], weight.shape[-1]])
    shape_diff = im_shape - kernel_shape 
    pad = np.sign(shape_diff) * np.abs(shape_diff)//2 
    odd_bias = np.abs(shape_diff)%2
    oddity_old = kernel_shape%2
    pad_list = [pad[-1] + odd_bias[-1] * oddity_old[-1],\
    pad[-1] + odd_bias[-1] * (1-oddity_old[-1]),\
    pad[-2] + odd_bias[-2] * oddity_old[-2],\
    pad[-2] + odd_bias[-2] * (1-oddity_old[-2])] # starting from the last dimension and moving forward, (padding_left,padding_right, padding_top,padding_bottom)'
    
    spectral_weight =   rfftshift(
                            fft.rfft2(
                                fft.ifftshift(
                                    tf.pad(weight, pad_list), 
                                    dim = [-2,-1]
                                ), 
                                norm = norm
                            )
                        )
    
    # the discrete approximation of continuous convolution in spatial domain differs from the spectral implementation for even dimensions, if conv_like_cnn, we use spatial approach
    if conv_like_cnn:
        if im_shape[-2]%2 == 0:
            spectral_weight[...,0,:] *=2
        if im_shape[-1]%2 == 0:
            spectral_weight[...,:,0] *=2

    if not ksize is None:
        ksize = np.array(ksize)

        spectral_weight = symmetric_padding(spectral_weight, im_shape, ksize)
    return spectral_weight

def spectral_to_spatial(weight, im_shape, odd = True, norm = 'forward', conv_like_cnn=False):
        weight = weight.clone()
        #maybe this doesn't work yet
        if conv_like_cnn:
            if im_shape[-2]%2 == 0:
                weight[...,0,:] *=0.5
            if im_shape[-1]%2 == 0:
                weight[...,:,0] *=0.5

        ksize2 = weight.shape[-1]*2 - (2 - odd) #odd = True: n*2 - 1; odd = False: n*2 - 2 

        kernel_shape = np.array([weight.shape[-2], ksize2])
        multiplier_padded = symmetric_padding(weight, kernel_shape, im_shape) 

        return fft.fftshift(
                    fft.irfft2(
                        irfftshift(multiplier_padded),
                        s=tuple(im_shape), norm = norm
                    ),
                    dim = [-2,-1]
                )
  
def conv_to_spectral(conv, im_shape, parametrization='spectral', norm='forward',\
                     in_shape=None, out_shape=None, conv_like_cnn = True,
                     ksize=None,
                     stride_trigo = False,
                     stride = None):
    im_shape = np.array(im_shape)
    weight = conv.weight
    weight = torch.flip(conv.weight, dims = [-2,-1])
    weight = torch.permute(weight, (1,0,2,3)) #torch.conv2d performs a cross-correlation, i.e., convolution with flipped weight
    if norm == 'forward':
        weight*=np.prod(im_shape)
    elif norm == 'ortho':
        weight*= np.sqrt(np.prod(im_shape))
    
    if parametrization=='spectral':
        weight = spatial_to_spectral(weight, im_shape, norm=norm, conv_like_cnn=conv_like_cnn, ksize=ksize)

    odd = ((im_shape[-1]%2) == 1)
    
    if stride is None:
        stride = conv.stride

    return SpectralConv2d(
            conv.in_channels, conv.out_channels,
            parametrization=parametrization,
            weight=weight, stride = stride, odd = odd,
            in_shape = in_shape, out_shape = out_shape,
            conv_like_cnn = conv_like_cnn,
            norm=norm,
            stride_trigo = stride_trigo
            )
        
    
    
                

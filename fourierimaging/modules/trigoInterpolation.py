import numpy as np
import torch
import torch.fft as fft
import torchvision.transforms.functional as TF
import torch.nn.functional as tf
import torch.nn as nn

def rfftshift(x):
    ##Returns a shifted version of a matrix obtained by torch.fft.rfft2(x), i.e. the left half of torch.fft.fftshift(torch.fft.fft2(x))
    # Real FT shifting thus consists of two steps: 1d-Shifting the last dimension and flipping in second to last dimension and taking the complex conjugate,
    # to get the frequency order [-N,...,-1,0]
    return torch.conj(torch.flip(fft.fftshift(x, dim=-2), dims=[-1]))

def irfftshift(x):
    ##Inverts rfftshift: Returns a non-shifted version of the left half of torch.fft.fftshift(torch.fft.fft2(x)), i.e. torch.fft.rfft2(x)
    return torch.conj(torch.flip(fft.ifftshift(x, dim=-2), dims=[-1]))

def symmetric_padding(xf_old, im_shape_old, im_shape_new):
    #xf_old has to be a shifted rfft2 matrix
    add_shape = np.array(xf_old.shape[:-2])

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
# - additional parameter 'parameterization' = {'spectral', 'spatial'}: determines wether the optimization is done in spectral (original) or spatial (addition) domain
# - additional parameters 'ksize1' and 'ksize2': determines kernel height and width if parametrization=='spatial'
# - additional parameter 'out_shape' = [int, int]: determines height and width of output. If not chosen, output shape will be the same as input shape
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, weight=None,\
                 out_shape=None, in_shape=None, parametrization = 'spectral',\
                 ksize1 = 1, ksize2 = 1,\
                 stride = 1, norm = 'forward',\
                 odd = True
                ):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = out_shape 
        self.in_shape = in_shape
        self.parametrization = parametrization
        self.stride = stride
        self.norm = norm
        self.odd = odd

        self.scale = (1 / (in_channels * out_channels))
    
        #Initialize weight according to chosen parametrization
        if parametrization == 'spectral':
            if weight is None:              
                weight = torch.rand(in_channels, out_channels,\
                                     ksize1, ksize2//2+1,\
                                     dtype=torch.cfloat)
                self.odd = ksize2%2
                weight[:,:,0,0].imag = 0.  
            self.ksize1 = weight.shape[-2]
            self.ksize2 = weight.shape[-1]*2 - (2 - self.odd)                 
        elif parametrization == 'spatial':
            if weight is None:        
                weight = self.scale*torch.rand(in_channels, out_channels, ksize1, ksize2)
            self.ksize1 = weight.shape[-2]
            self.ksize2 = weight.shape[-1]
        self.weight = nn.Parameter(weight)
        
    # Complex multiplication (original)
    def compl_mul2d(self, input, weight):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weight)

    def forward(self, x):
        x_shape = np.array(x.shape[-2:])

        if self.in_shape == None:
            im_shape_old = x_shape # variable spatial support of kernel in case of spatial parametrization
        else:
            im_shape_old = np.array(self.in_shape) # close to fixed spatial support in case of spatial and spectral parametrization

        if self.out_shape == None:
            im_shape_new = x_shape
        else:
            im_shape_new = np.array(self.out_shape)

        if self.parametrization=='spectral':
            kernel_shape = np.array([self.ksize1, self.ksize2])
            multiplier_padded = symmetric_padding(self.weight, kernel_shape, im_shape_new) 
        elif self.parametrization=='spatial':
            multiplier = spatial_to_spectral(self.weight, im_shape_old, norm=self.norm)
            # spatial zero-padding to match image shape, then ifftshift to align center, then rfft2 and rfftshift to get multiplier, maybe this can be optimized by using the 's' parameter for rfft2
            multiplier_padded = symmetric_padding(multiplier,\
                                                  im_shape_old,\
                                                  im_shape_new)
        #convolution is implemented by elementwise multiplication of rfft-coefficients (only for odd dimensions, for even dimensions we interpolate to the next higher odd dimension)
        #this could be optimized by checking dimensions first
        x_ft_padded = symmetric_padding(rfftshift(fft.rfft2(x, norm = self.norm)), x_shape, im_shape_new)

        # Return to physical space after correcting dimension if desired dimension is even
        output = fft.irfft2(irfftshift(self.compl_mul2d(x_ft_padded, multiplier_padded),\
                                      ), 
                            norm = self.norm,\
                            s=tuple(im_shape_new))
        if sum(self.stride) > 1:
            output = output[...,0::self.stride[0], 0::self.stride[1]]
        return output


def spatial_to_spectral(weight, im_shape, norm='forward'):
    kernel_shape = np.array([weight.shape[-2], weight.shape[-1]])
    shape_diff = im_shape - kernel_shape 
    pad = np.sign(shape_diff) * np.abs(shape_diff)//2 
    odd_bias = np.abs(shape_diff)%2
    oddity_old = kernel_shape%2
    pad_list = [pad[-1] + odd_bias[-1] * oddity_old[-1],\
    pad[-1] + odd_bias[-1] * (1-oddity_old[-1]),\
    pad[-2] + odd_bias[-2] * oddity_old[-2],\
    pad[-2] + odd_bias[-2] * (1-oddity_old[-2])] # starting from the last dimension and moving forward, (padding_left,padding_right, padding_top,padding_bottom)'
    
    return rfftshift(fft.rfft2(fft.ifftshift(tf.pad(weight, pad_list), dim = [-2,-1]), norm = norm))

def spectral_to_spatial(weight, im_shape, odd = True, norm = 'forward'):
        ksize2 = weight.shape[-1]*2 - (2 - odd) #odd = True: n*2 - 1; odd = False: n*2 - 2 

        kernel_shape = np.array([weight.shape[-2], ksize2])
        multiplier_padded = symmetric_padding(weight, kernel_shape, im_shape) 

        return fft.fftshift(\
                    fft.irfft2(irfftshift(multiplier_padded),\
                              s=im_shape, norm = norm),\
                    dim = [-2,-1])
  
def conv_to_spectral(conv, im_shape, parametrization='spectral', norm='forward'):
    im_shape = np.array(im_shape)
    weight = conv.weight
    weight = torch.flip(conv.weight, dims = [-2,-1])
    weight = torch.permute(weight, (1,0,2,3))*np.prod(im_shape) #torch.conv2d performs a cross-correlation, i.e., convolution with flipped weight
    
    if parametrization == 'spectral':
        weight = spatial_to_spectral(weight, im_shape, norm)
    odd = ((im_shape[-1]%2) == 1)

    return SpectralConv2d(conv.in_channels, conv.out_channels,\
                   parametrization=parametrization,\
                   weight=weight, stride = conv.stride, odd = odd)
        
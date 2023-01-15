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
    def __init__(self, in_channels, out_channels, out_shape=None, in_shape=None, parametrization = 'spectral', modes1=None, modes2=None, ksize1=None, ksize2=None, norm = 'forward'):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.parametrization = parametrization
        self.norm = norm

        self.scale = (1 / (in_channels * out_channels))
    
        #Initialize weights according to chosen parametrization
        if parametrization == 'spectral':
            if modes1==None | modes2==None:
                raise ValueError('To use spectral parametrization, please select modes1 and modes2. Currently modes1='+str(modes1) + 'and modes2=' +str(modes2))
            self.modes1 = modes1
            self.modes2 = modes2
            self.weights = torch.Parameter(torch.rand(in_channels, out_channels, (self.modes1-1)*2, self.modes2, dtype=torch.cfloat)) # the 2-D real FFT of a real-valued kernel with an odd number of sampling points
        elif parametrization == 'spatial':
            if ksize1==None | ksize2==None:
                raise ValueError('To use spatial parametrization, please select ksize1 and ksize2. Currently ksize1='+str(ksize1) + 'and modes2=' +str(ksize2))
            self.ksize1 = ksize1
            self.ksize2 = ksize2
            self.weights = nn.Parameter(self.scale*torch.rand(in_channels, out_channels, self.ksize1, self.ksize2))

        self.out_shape = out_shape 
        self.in_shape = in_shape

    # Complex multiplication (original)
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_shape = np.array(x.shape[-2:])

        if self.in_shape == None:
            im_shape_old = x_shape
        else:
            im_shape_old = np.array(self.in_shape)

        if self.out_shape == None:
            im_shape_new = x_shape
        else:
            im_shape_old = np.array(self.out_shape)

        if self.parametrization=='spectral':
            multiplier = self.weights
        elif self.parametrization=='spatial':
            kernel_shape = np.array([self.ksize1, self.ksize2])
            shape_diff = im_shape_old - kernel_shape
            pad = np.sign(shape_diff) * np.abs(shape_diff)//2
            odd_bias = np.abs(shape_diff)%2
            oddity_old = kernel_shape%2
            pad_list = [pad[-1] + odd_bias[-1] * oddity_old[-1],\
            pad[-1] + odd_bias[-1] * (1-oddity_old[-1]),\
            pad[-2] + odd_bias[-2] * oddity_old[-2],\
            pad[-2] + odd_bias[-2] * (1-oddity_old[-2])] #'starting from the last dimension and moving forward, (padding_left,padding_right, padding_top,padding_bottom)'
            
            # spatial zero-padding to match image shape, then ifftshift to align center, then rfft2 and rfftshift to get multiplier, maybe this can be optimized by using the 's' parameter for rfft2
            multiplier = rfftshift(fft.rfft2(fft.ifftshift(torch.pad(self.weights, pad_list), dim = [-2,-1]), norm = self.norm))
        
        #convolution is implemented by elementwise multiplication of rfft-coefficients (only for odd dimensions, for even dimensions we interpolate to the next higher odd dimension)
        #this could be optimized by checking dimensions first
        x_ft_padded = symmetric_padding(rfftshift(fft.rfft2(x)), x_shape, im_shape_new + (1 - im_shape_new%2)) #odd dimensions
        multiplier_padded = symmetric_padding(multiplier, im_shape_old, im_shape_new + (1 - im_shape_new%2)) #odd dimensions

        # Return to physical space after correcting dimension if desired dimension is even
        output = fft.irfft2(irfftshift(symmetric_padding(self.compl_mul2d(x_ft_padded, multiplier_padded),  im_shape_new + (1 - im_shape_new%2), im_shape_new)), norm = self.norm)

        return output

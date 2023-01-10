import numpy as np
import torch
import torch.fft as fft
import torchvision.transforms.functional as TF
import torch.nn.functional as tf
import torch.nn as nn

class TrigonometricResize_2d:
    """Resize 2d image with trigonometric interpolation"""

    def __init__(self, shape, norm = 'forward'):
        self.shape = shape
        self.norm = norm
    
    def __call__(self, x):

        ## Compute a symmetric Fourier Transform for easier interpolation
        old_shape = np.array(x.shape[-2:])

        # for odd dimensions, the fft is already symmetric
        oldft_shape = old_shape + (1 - old_shape%2)

        xf = torch.zeros(x.shape[0], x.shape[1], oldft_shape[-2], oldft_shape[-1], dtype=torch.cfloat)
        xf[:,:,:old_shape[-2], :old_shape[-1]] = fft.fftshift(fft.fft2(x, norm = self.norm)) #shift for easier handling

        # for even dimensions, the coefficients corresponding to the nyquist frequency are split symmetrically
        if old_shape[-2] < oldft_shape[-2]:
            nyquist = xf[:,:,0,:]/2
            xf[:,:,0,:] = nyquist
            xf[:,:,-1,:] = nyquist #this is equivalent to taking the complex conjugate of the flipped nyquist row
        
        if old_shape[-1] < oldft_shape[-1]:
            nyquist = xf[:,:,:,0]/2
            xf[:,:,:,0] = nyquist
            xf[:,:,:,-1] = nyquist #this is equivalent to taking the complex conjugate of the flipped nyquist column
        new_shape = np.array(self.shape)
        #for even dimensions, first create a finer but symmetric Fourier transform
        newft_shape = new_shape + (1-new_shape%2)

        pad = ((newft_shape - oldft_shape)/2).astype(int) #the difference between both ft shapes is always even
        pad_list = [pad[1], pad[1], pad[0], pad[0]] #according to torch.nn.functional.pad documentation: 'starting from the last dimension and moving forward'
    
        xf_pad = tf.pad(xf, pad_list)
        if new_shape[-2] < newft_shape[-2]:
            xf_pad[:,:,0,:] = xf_pad[:,:,0,:]*2
        
        if new_shape[-1] < newft_shape[-1]:
            xf_pad[:,:,:,0] = xf_pad[:,:,:,0]*2
        x_inter = fft.ifft2(fft.ifftshift(xf_pad[:,:,:new_shape[-2],:new_shape[-1]]), norm=self.norm).type(x.dtype)
        return x_inter

# This is a modified version of https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
# Modifications and additions are as follows:
# - additional parameter 'parameterization' = {'spectral', 'spatial'}: determines wether the optimization is done in spectral (original) or spatial (addition) domain
# - additional parameters 'ksize1' and 'ksize2': determines kernel height and width if parametrization=='spatial'
# - additional parameter 'output_shape' = [int, int]: determines height and width of output
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape, ksize1, ksize2, parametrization = 'spectral'):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.parametrization = parametrization

        self.scale = (1 / (in_channels * out_channels))
        
        self.ksize1 = ksize1
        self.ksize2 = ksize2
        if parametrization == 'spectral':
            self.negative_height = ksize1//2+1
            self.negative_width = ksize2//2+1
            self.positive_height = ksize1//2
            self.positive_width = ksize2-ksize2//2-1
            negative_weights = self.scale * torch.rand(in_channels, out_channels, self.negative_height, self.negative_width, dtype=torch.cfloat)
            negative_weights[:,:,-1,-1].imag = 0 #weights should be the Fourier-coefficients of a real-valued kernel
            self.negative_weights = nn.Parameter(negative_weights)
            self.positive_weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.positive_height, self.positive_width, dtype=torch.cfloat))
        elif parametrization == 'spatial':
            self.weights = nn.Parameter(self.scale*torch.rand(in_channels, out_channels, self.ksize1, self.ksize2))

        self.output_shape = output_shape 

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        #modified: for easier handling we use fft2 instead of rfft2, even though rfft2 would be sufficient
        x_ft = fft.fft2(x)

        #Compute Fourier coefficients of kernel (odd dimensions, not shifted)
        if self.parametrization=='spectral':
            fkernel = torch.zeros(self.in_channels, self.out_channels, self.ksize1 + (1-self.ksize1%2), self.ksize2 + (1-self.ksize2%2), dtype = torch.cfloat) #odd dimensions
            fkernel[:,:,:self.negative_height, :self.negative_width] = self.negative_weights
            fkernel[:,:,:self.positive_height, -self.negative_height:] = self.positive_weights
            fkernel = fkernel + torch.conj(torch.flip(fkernel, dims=[-2,-1]))

            # zero-padding for correct output shape 
            # HERE

            fkernel = fft.ifftshift(fkernel)


        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft. irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


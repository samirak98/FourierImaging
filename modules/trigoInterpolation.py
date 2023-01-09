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
# - additional parameter 'output_like' = {'input', 'kernel', 'fixed'}: determines height and width of output in the following way
                                                                    # - 'input' (original): output shape is given by input shape (zero-padding or cutting off frequencies if necessary)
                                                                    # - 'kernel' (addition): output shape is given by modes1 and modes2 (no zero-padding or cutting off frequencies)
                                                                    # - 'fixed' (addition): output shape is given by parameter 'output_shape' (zero-padding or cutting off frequencies if necessary)
# - additional parameter 'output_shape' = [int, int]: determines height and width of output if output_like=='fixed'
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, parametrization = 'spectral', modes1=None, modes2=None, ksize1=None, ksize2=None, output_like = 'input'):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.parametrization = parametrization

        self.scale = (1 / (in_channels * out_channels))

        if parametrization == 'spectral':
            if modes1==None | modes2==None:
                print('To use a spectral parametrization, please specify modes1 and modes2')
                return
            self.modes1 = modes1 
            self.modes2 = modes2
            self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
            self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        elif parametrization == 'spatial':
            if ksize1==None | ksize2==None:
                print('To use a spatial parametrization, please specify ksize1 and ksize2')
                return
            self.ksize1 = ksize1
            self.ksize2 = ksize2
            self.weights = nn.Parameter(self.scale*torch.rand(in_channels, out_channels, self.ksize1, self.ksize2))

        self.output_like = output_like
        ## here

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft. irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


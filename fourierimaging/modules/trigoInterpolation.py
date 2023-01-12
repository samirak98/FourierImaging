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
    x = torch.conj(torch.flip(fft.fftshift(x, dim=-1), dims=[-2]))
    return x

def irfftshift(x):
    ##Inverts rfftshift: Returns a non-shifted version of the left half of torch.fft.fftshift(torch.fft.fft2(x)), i.e. torch.fft.rfft2(x)
    x = torch.conj(torch.flip(fft.ifftshift(x, dim=-1), dims=[-2]))

class TrigonometricResize_2d:
    """Resize 2d image with trigonometric interpolation"""

    def __init__(self, shape, norm = 'forward', check_comp=False):
        self.shape = shape
        self.norm = norm
        self.check_comp = check_comp
    
    def __call__(self, x):
        add_dims = np.array(x.shape[:-2])
        old_shape = np.array(x.shape[-2:])
        new_shape = np.array(self.shape)

        if torch.is_complex(x):
            # if we work with complex valued functions, trigonometric interpolations is done by simple zero-padding of the Fourier coefficients
            pad = (new_shape - old_shape)//2
            odd_bias = pad%2
            pad_list = [pad[-1]+odd_bias[-1], pad[-1], pad[-2]+odd_bias[-2], pad[-2]] #'starting from the last dimension and moving forward, (padding_left,padding_right, padding_top,padding_bottom)'
            
            xf = fft.fftshift(fft.fft2(x, norm = self.norm), dim=[-2,-1])
            xf_pad = tf.pad(xf, pad_list)
            x_inter = fft.ifft2(fft.ifftshift(xf_pad, dim=[-2,-1]), norm=self.norm)
        else:
        # for real valued functions, the coefficients have to be Hermitian symmetric
            xf = rfftshift(fft.rfft2(x, pad_list))
            oldft_height = old_shape[-2] + (1-old_shape[-2]%2)
            oldft_width = old_shape[-1]//2 + 1

            xf = torch.zeros(np.concatenate(add_dims, oldft_height, oldft_width),  dtype=torch.cfloat, device=x.device)
            xf[...,:old_shape[-2]] = rfftshift(fft.rfft2(x, norm = self.norm)) #for odd dimensions, this already represents a set of Herm.sym. coefficients
            
            # for even dimensions, the coefficients corresponding to the nyquist frequency are split symmetrically
            if old_shape[-2]%2 == 0:
                xf[...,0 ,:]*= 0.5 # nyquist row /2
                xf[...,-1 ,:] = xf[...,0 ,:] # nyquist_row/2
                
            if old_shape[-1] < oldft_shape[-1]:
                xf[..., :,  0]  *= 0.5#   = nyquist_col[:,:, 1:-1]/2
                xf[..., :, -1] = xf[..., : ,  0] #nyquist_col[:,:, 1:-1]/2


            

        ## Compute a symmetric Fourier Transform for easier interpolation
        old_shape = np.array(x.shape[-2:])

        # for odd dimensions, the fft is already symmetric
        oldft_shape = old_shape + (1 - old_shape%2)
        xf = torch.zeros(x.shape[0], x.shape[1], oldft_shape[-2], oldft_shape[-1], dtype=torch.cfloat, device=x.device)
        xf[:,:,:old_shape[-2], :old_shape[-1]] =  fft.fftshift(fft.fft2(x, norm = self.norm))

        # for even dimensions, the coefficients corresponding to the nyquist frequency are split symmetrically
        if old_shape[-2] < oldft_shape[-2]:
            xf[...,0 ,:]*= 0.5 # nyquist row /2
            xf[...,-1 ,:] = xf[...,0 ,:] # nyquist_row/2
            
        if old_shape[-1] < oldft_shape[-1]:
            xf[..., :,  0]  *= 0.5#   = nyquist_col[:,:, 1:-1]/2
            xf[..., :, -1] = xf[..., : ,  0] #nyquist_col[:,:, 1:-1]/2

        new_shape = np.array(self.shape)

        if self.check_comp:
            self.check_symmetry(xf, old_shape=old_shape)
        
        ## Trigonometric interpolation: Zero-Padding/Cut-Off of the symmetric (odd-dimensional) Fourier transform, if needed convert to even dimensions
        #Zero-padding/cut-off to odd_dimension such that desired_dimension <= odd_dimension <= desired_dimension + 1
        newft_shape = new_shape + (1-new_shape%2)
        pad = (newft_shape - oldft_shape)//2 #the difference between both ft shapes is always even
        pad_list = [pad[1], pad[1], pad[0], pad[0]] #'starting from the last dimension and moving forward, (padding_left,padding_right, padding_top,padding_bottom)'
        xf_pad = tf.pad(xf, pad_list)

        #if desired dimension is even, reverse the nyquist splitting
        if new_shape[-2] < newft_shape[-2]:
            xf_pad[...,0 ,:]*= 2 #  = nyquist_row[:,:, 1:-1]/2
                      
        if new_shape[-1] < newft_shape[-1]:
            xf_pad[...,: ,  0]  *= 2#   = nyquist_col[:,:, 1:-1]/2

        x_inter = fft.ifft2(fft.ifftshift(xf_pad[:,:,:new_shape[-2],:new_shape[-1]]), norm=self.norm)
        
        if self.check_comp:
            imag_norm = torch.norm(x_inter.imag, p=float("Inf"))
            if  imag_norm > 1e-5:
                print('The imaginary part of the image is unusual high, norm: ' +str(imag_norm))
                print(old_shape)

        ## The interpolated function should be real-valued
        x_inter = x_inter.real
        return x_inter

    def check_symmetry(self, x, old_shape=None, threshold=1e-5):
        ## Helper function to check Hermitian symmetry of an odd-dimensioned matrix of Fourier coefficients. 
        # Symmetry is fulfilled iff the coefficients correspond to a function which attains only real values at the considered sampling points
        x_flip = torch.flip(x, dims=[-2,-1])
        symmetry_check = x - torch.conj(x_flip)
        symmetry_norm = torch.norm(symmetry_check, p=float("Inf"))
        if  symmetry_norm > threshold:
            print('Not symmetric: (norm: ' + str(symmetry_norm) + ' for old shape: ' + str(old_shape))


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


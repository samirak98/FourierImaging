import numpy as np
import torch
import torch.fft as fft
import torchvision.transforms.functional as TF
import torch.nn.functional as tf

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
        print(pad_list)

        xf_pad = tf.pad(xf, pad_list)
        if new_shape[-2] < newft_shape[-2]:
            xf_pad[:,:,0,:] = xf_pad[:,:,0,:]*2
        
        if new_shape[-1] < newft_shape[-1]:
            xf_pad[:,:,:,0] = xf_pad[:,:,:,0]*2
        x_inter = fft.ifft2(fft.ifftshift(xf_pad[:,:,:new_shape[-2],:new_shape[-1]]), norm = self.norm)
        
            


        return xf, xf_pad, x_inter
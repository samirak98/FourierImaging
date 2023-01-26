import numpy as np
import scipy as scp
import skimage as sk
import matplotlib.pyplot as plt
import matplotlib as mpl
#%%
#img_path = '../../../datasets/CUBirds200/train/BELTED KINGFISHER/084.jpg'
img_path = '../../../datasets/CUBirds200/train/RED WISKERED BULBUL/029.jpg'
I = plt.imread(img_path)
I = I[0:I.shape[0]//2-1, 0:I.shape[0]//2-1, :]
I = I/255.

#%%
def bb(n, m):
    B = np.zeros((n,n), dtype=bool)
    mm = m//2
    nn = n//2
    B[:,nn+mm] = True
    B[:,nn-mm] = True
    B[nn+mm,:] = True
    return B

#%%
k_low = I.shape[0]//4 + (1-(I.shape[0]//2)%2)
k_mid = I.shape[0]//2 + (1-(I.shape[0]//2)%2)
k_high = I.shape[0]

#%%
filter_name = 'sobel'
colormap = 'coolwarm'

if filter_name == 'sobel':
    k_size = 11
    kernel = np.zeros((k_size, k_size))
    
    for i in range(k_size):
        for j in range(k_size):
            ii = i - k_size//2
            jj = j - k_size//2
            
            nom = ii**2 + jj**2
            if np.abs(nom)>0:
                kernel[i,j] = ii / nom
    noise_lvl = 0.
elif filter_name == 'mean':
    k_size = 5
    var = 2
    #kernel = np.ones((k_size,k_size))
    s = np.linspace(-(k_size - 1) * 0.5, (k_size - 1) * 0.5, k_size)
    gauss = np.exp(-0.5 * s**2 / var**2)
    kernel = np.outer(gauss, gauss)

    
    noise_lvl = 0.05
            
kernel *= 1/np.linalg.norm(kernel)
true_kernel = np.pad(kernel, (k_mid-k_size)//2)


#%%
plt.close('all')
fig, ax = plt.subplots(3,5)

#%% plot imgs
for i in range(3):
    loc_ax = ax[i,2]
    
    if i==2:
        I_high = I.copy()
        I_high += np.random.normal(0, noise_lvl, size = I_high.shape)
        J = I_high
    elif i==1:
        I_mid = sk.transform.resize(I, [k_mid, k_mid, 3])
        I_mid += np.random.normal(0, noise_lvl, size = I_mid.shape)
        J = I_mid
    else:
        I_low = sk.transform.resize(I, [k_low, k_low, 3])
        I_low += np.random.normal(0, noise_lvl, size = I_low.shape)
        J = I_low
    loc_ax.imshow(J)
    loc_ax.axis('off')
    
#%% plot kernels
for i in range(3):
    loc_ax = ax[i,3]
    
    if i==2:
        B = bb(k_high, k_mid)
        k = np.pad(kernel, (k_high-k_size)//2)
    elif i==0:
        k = np.pad(kernel, (k_low-k_size)//2)
        B = bb(k_high, k_mid)
    else:
        k = np.pad(kernel, (k_mid-k_size)//2)
        B = bb(k_high, k_mid)
        
    loc_ax.imshow(k, cmap=mpl.colormaps[colormap])
    loc_ax.axis('off')
    
#%% plot kernels
for i in range(3):
    loc_ax = ax[i,4]
    
    if i==2:
        k = np.pad(kernel, (k_high-k_size)//2)
        k = np.fft.ifftshift(k)
        J = np.transpose(I_high, axes=[2,0,1])*1.
    elif i==0:
        k = np.pad(kernel, (k_low-k_size)//2)
        k = np.fft.ifftshift(k)
        J = np.transpose(I_low, axes=[2,0,1])*1.
    else:
        k = np.pad(kernel, (k_mid-k_size)//2)
        k = np.fft.ifftshift(k)
        J = np.transpose(I_mid, axes=[2,0,1])*1.
                 
    J = np.fft.ifft2(
            np.fft.fft2(J) * 
            np.fft.fft2(k[np.newaxis, :,:])
            )
    J = np.real(J)
    J = (J - J.min(axis=(1,2))[:,np.newaxis,np.newaxis])
    J *= 1/J.max(axis=(1,2))[:,np.newaxis,np.newaxis]
    J = np.transpose(J, axes=[1,2,0])
        
    loc_ax.imshow(J)
    loc_ax.axis('off')
#%% get spectral kernels
fk = np.fft.fft2(true_kernel)
fk = np.fft.fftshift(fk)
pad_low = (k_mid - k_low)//2
fk_low = fk[(pad_low+1):-(pad_low-1), (pad_low+1):-(pad_low-1)]
fk_low = np.fft.ifftshift(fk_low)
kernel_low = np.fft.ifft2(fk_low).real
# up scale
fk_high = np.pad(fk, (k_high-k_mid)//2)
fk_high = np.fft.ifftshift(fk_high)
kernel_high = np.fft.ifft2(fk_high).real

#%% plot spectral padded kernels
for i in range(3):
    loc_ax = ax[i,1]
    
    if i==1:
        k = true_kernel
    elif i==0:
        k = kernel_low
    else:
        k = kernel_high
        
    loc_ax.imshow(k, cmap=mpl.colormaps[colormap])
    loc_ax.axis('off')
    
#%% plot spectral conv
for i in range(3):
    loc_ax = ax[i,0]
    
    if i==1:
        k = true_kernel
        k = np.fft.ifftshift(k)
        J = I_mid
    elif i==0:
        k = kernel_low
        k = np.fft.ifftshift(k)
        J = I_low
    else:
        k = kernel_high
        k = np.fft.ifftshift(k)
        J = I_high
    J = np.transpose(J, axes=[2,0,1])*1.  
    J = np.fft.ifft2(
            np.fft.fft2(J) * 
            np.fft.fft2(k[np.newaxis, :,:])
            )
    J = np.real(J)
    J = (J - J.min(axis=(1,2))[:,np.newaxis,np.newaxis])
    J *= 1/J.max(axis=(1,2))[:,np.newaxis,np.newaxis]
    J = np.transpose(J, axes=[1,2,0])
        
    loc_ax.imshow(J)
    loc_ax.axis('off')
    
#%%
save = True
if save:
    plt.tight_layout(pad=0.2)
    plt.savefig('res-conv-' + filter_name + '.pdf')


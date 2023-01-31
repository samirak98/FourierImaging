import matplotlib.pyplot as plt
from matplotlib import rc
from cycler import cycler
import csv
import numpy as np
#%% visulaize data
plt.close('all')
#plt.style.use(['ggplot'])
plt.style.use(['seaborn-whitegrid'])
default_cycler = (cycler(color=['xkcd:sky', 
                                'xkcd:apple','olive','xkcd:grapefruit',\
                                'xkcd:muted blue','peru','tab:pink',\
                                'deeppink', 'steelblue', 'tan', 'sienna',\
                                'olive', 'coral']))
rc('font',**{'family':'lmodern','serif':['Times'],'size':14})
rc('text', usetex=True)
rc('lines', linewidth=2, linestyle='-')
rc('axes', prop_cycle=default_cycler)



fig,ax = plt.subplots(1,2,figsize=(8.27/1.5,11.69/4), sharey=True)
accs = []

#fnames = ['results/CUB200-spectral-2.csv', 'results/CUB200-circular.csv']
fnames = ['results/FMNIST-spectral-3.csv', 'results/FMNIST-3.csv']
for j,fname in enumerate(fnames):
    with open(fname, 'r') as f:
        reader = csv.reader(f, lineterminator = '\n')
        old_data = None
        ax_idx=-1
        for i,row in enumerate(reader):
            if i == 0:
                sizes = np.array(row[2:], dtype=np.float64)
                idx = np.argsort(sizes)
            else:
                if old_data != row[0]:
                    #ax = fig.add_subplot(ax_idx)#plt.figure()
                    ax_idx+=1
                    old_data = row[0]
                    ax[ax_idx].set_title('Data sizing: ' + row[0])
                
                if j == 0:
                    zorder = 5
                    if row[1] == 'NONE':
                        name = 'FNO'
                        zorder = 10
                    elif row[1] == 'TRIGO':
                        name = 'Trigonometric\n Interpolation'
                    elif row[1] == 'BILINEAR':
                        name = 'Bilinear\n Interpolation'
                        
                    vals = np.array(row[2:], dtype=np.float64)
                    ax[ax_idx].plot(sizes[idx], vals[idx], label = name, zorder = zorder)
                    ax[ax_idx].set_xlabel('Input Image Size')
                    
                    if ax_idx==0:
                        ax[ax_idx].set_ylabel('Test Accuracy')
                else:
                    if row[1] == 'NONE':
                        name = 'CNN'
                        vals = np.array(row[2:], dtype=np.float64)
                        ax[ax_idx].plot(sizes[idx], vals[idx], label = name)
                #ax[ax_idx].xaxis.set_ticks([3,7,14,21,28])
                #ax[ax_idx].xaxis.set_ticks(np.arange(3,214,50))
                if ax_idx==0:
                    pass
                    ax[ax_idx].legend(#loc='lower right',
                                      fontsize=14)
            
#%%
save = True
if save:
    plt.tight_layout(pad=0.1)
    #plt.savefig('CUB200-Interpolation.pdf')
    plt.savefig('FMNIST-Interpolation.pdf')
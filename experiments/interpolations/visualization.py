import matplotlib.pyplot as plt
from matplotlib import rc
from cycler import cycler
import csv
import numpy as np
#%% visulaize data
plt.close('all')
plt.style.use(['ggplot'])
default_cycler = (cycler(color=['grey', 
                                'xkcd:apple','xkcd:sky','xkcd:grapefruit',\
                                'xkcd:muted blue','peru','tab:pink',\
                                'deeppink', 'steelblue', 'tan', 'sienna',\
                                'olive', 'coral']))
rc('font',**{'family':'lmodern','serif':['Times'],'size':10})
rc('text', usetex=True)
rc('lines', linewidth=1, linestyle='-')
rc('axes', prop_cycle=default_cycler)



fig,ax = plt.subplots(1,4,figsize=(8.27,11.69/5), sharey=True)
accs = []
with open('results/CUB200-normal.csv', 'r') as f:
    reader = csv.reader(f, lineterminator = '\n')
    old_data = None
    ax_idx=-1
    for i,row in enumerate(reader):
        if i == 0:
            sizes = np.array(row[2:], dtype=np.float64)
        else:
            if old_data != row[0]:
                #ax = fig.add_subplot(ax_idx)#plt.figure()
                ax_idx+=1
                old_data = row[0]
                ax[ax_idx].set_title('Data sizing: ' + row[0])
                
            vals = np.array(row[2:], dtype=np.float64)
            ax[ax_idx].plot(sizes, vals, label=row[1])
            #ax[ax_idx].xaxis.set_ticks([3,7,14,21,28])
            ax[ax_idx].xaxis.set_ticks(np.arange(3,214,50))
            if ax_idx==0:
                ax[ax_idx].legend()
 
#%%
save = False
if save:
    plt.tight_layout(pad=0.1)
    plt.savefig('FMNIST-Interpolation.pdf')
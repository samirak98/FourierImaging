import matplotlib.pyplot as plt
from matplotlib import rc
from cycler import cycler
import csv
import numpy as np
#%% visulaize data
plt.close('all')
plt.style.use(['seaborn'])
default_cycler = (cycler(color=['xkcd:apple','xkcd:sky','xkcd:coral',\
                                'xkcd:light gold','peru','tab:pink',\
                                'deeppink', 'steelblue', 'tan', 'sienna',\
                                'olive', 'coral']))
rc('font',**{'family':'lmodern','serif':['Times'],'size':14})
rc('text', usetex=True)
rc('lines', linewidth=1, linestyle='-')
rc('axes', prop_cycle=default_cycler)


fig,ax = plt.subplots(1,4,figsize=(8.27,11.69/5))
accs = []
with open('results/MNIST-save.csv', 'r') as f:
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
            ax[ax_idx].legend()
 
#%%
save = False
if save:
    plt.tight_layout(pad=0.1)
    plt.savefig('MNIST-Interpolation.pdf')
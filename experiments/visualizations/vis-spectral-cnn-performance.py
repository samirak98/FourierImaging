import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from cycler import cycler

fname = 'spectral-cnn-perf.csv'
accs = {}

default_cycler = (cycler(color=['olive','xkcd:apple',
                                'xkcd:grapefruit','xkcd:sky',
                                'xkcd:muted blue','peru','tab:pink',
                                'deeppink', 'steelblue', 'tan', 'sienna',
                                'olive', 'coral']))

plt.style.use(['seaborn-whitegrid'])
rc('font',**{'family':'lmodern','serif':['Times'],'size':14})
rc('text', usetex=True)
rc('lines', linewidth=2, linestyle='-')
rc('axes', prop_cycle=default_cycler)

with open(fname, 'r') as f:
    reader = csv.reader(f, lineterminator = '\n')
    for i,row in enumerate(reader):
        path = row[0]
        idx = path[::-1].find('/')
        name = path[-idx:]
        
        idx = name[::-1].find('-')
        ksize = int(name[-idx:])
        
        if ksize>=10:
            param = name[:-6]
        else:
            param = name[:-4]
            
        if param in accs:
            accs[param].append([ksize, float(row[1])])
        else:
            accs[param] = [[ksize, float(row[1])]]
            
        
        
        print(param)
        print(ksize)
        
#%%
plt.close('all')
fig, ax = plt.subplots(figsize=(8.27/1.5,11.69/5))

for param in accs.keys():
    a = np.array(accs[param])
    
    name_start = param[:5]
    print(name_start)
    if name_start == 'cnn-2':
        name = 'CNN 28 x 28 to spectral'
    elif name_start == 'cnn-3':
        name = 'CNN 3 x 3 to spectral'
    elif name_start == 'cnn':
        name = 'CNN'
    elif name_start == 'spect':
        name = 'FNO'
        
    plt.plot(a[:,0], a[:,1], label=name, marker='o')
    
#ax.set_ylim([0.7,.9])
ax.set_xlabel('Kernel Size')
ax.set_ylabel('Test Accuracy')
    
plt.legend()

#%%
save = True
if save:
    plt.tight_layout(pad=0.2)
    plt.savefig('ksize.pdf')
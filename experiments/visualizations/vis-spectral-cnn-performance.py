import csv
import matplotlib.pyplot as plt
import numpy as np

fname = 'spectral-cnn-perf.csv'
accs = {}

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
fig, ax = plt.subplots()

for param in accs.keys():
    a = np.array(accs[param])
    plt.plot(a[:,0], a[:,1], label=param)
    
ax.set_ylim([0.7,.9])
    
plt.legend()
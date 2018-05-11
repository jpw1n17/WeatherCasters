import numpy
import matplotlib.pyplot as plt
import pandas as pd
import pickle
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

#====== Prepare data
dataset = pd.read_csv('../../data/train.csv', sep = ',', header = 1)
datalen = dataset.shape[1]
outputlen = 24
inputlen = datalen-outputlen

y = dataset[dataset.columns[4:28]]
y.columns = ['s1','s2','s3','s4','s5','w1','w2','w3','w4','k1','k2','k3','k4','k5','k6','k7','k8','k9','k10','k11','k12','k13','k14','k15']


##======Compare result distribution
n_bins=20
fig, axs = plt.subplots(1, 2, sharey=True, sharex=False, tight_layout=True)
# We can set the number of bins with the `bins` kwarg
axs[0].hist(y['w1'], bins=n_bins)
axs[0].set_ylabel('Frequency')
axs[0].set_xlabel('Probability')
axs[0].set_title('W1 (When) - current weather')
axs[1].hist(y['k3'], bins=n_bins)
#axs[1].set_ylabel('Frequency')
axs[1].set_xlabel('Probability')
axs[1].set_title('K3 (Kind) - Dry weather')
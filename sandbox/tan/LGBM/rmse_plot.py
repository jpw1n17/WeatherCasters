import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import os
import lightgbm as lgb
import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def load_object(path):
    with open(path, 'rb') as input:
        obj = pickle.load(input)
    return obj   


rmse_all = load_object('case2_full_rmse.pkl')
rms_folds = rmse_all[3]
avg_rms = []

for i in range(len(rms_folds)):
    avg_rms.append(np.mean(rms_folds[i]))
avg_rms
    
avg_result = np.mean(avg_rms)
plt.figure(2)
plt.axis([0, 24, 0, 8])
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('RMSE from Light GBM')
ax.set_xlabel('output index')
#ax.set_xticks(x)
ax.set_ylabel('RMSE')
ax.set_ylim([0,0.8])
ax.bar(list(range(1,25)),avg_rms)
#fig.show()
ticks = plt.xticks(np.arange(25), ['']+ list(range(1,25)))
ax.axhline(y=avg_result, color='r', linestyle='-')

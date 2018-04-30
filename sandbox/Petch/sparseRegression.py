from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir('D:\Learning\Semester 2\DM\CW 1')


# =============================================================================
# read data from csv
# =============================================================================
num_data = pd.read_csv('Data\gencsv_ep100_vec100_Spre.csv'
                        ,header=None,sep=',',error_bad_lines=False,encoding='utf-8')

## =============================================================================
## put data into variables
## =============================================================================
n_output = 24
n_datacol = len(num_data.columns)

n_input = n_datacol-n_output

y = num_data.loc[:,n_input:n_input+n_output-1]
x = num_data.loc[:,0:n_input-1]

# =============================================================================
# create model
# =============================================================================
gamma = 0.2
clf = Ridge(alpha = gamma)
clf.fit(x, y)
Ridge(alpha=gamma, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)

# =============================================================================
# train model
# =============================================================================

all_nrows = len(x)
tr_nrows = round(all_nrows*0.8) - 1

ytr = y[0:tr_nrows]
xtr = x[0:tr_nrows]

yts = y[tr_nrows:all_nrows]
xts = x[tr_nrows:all_nrows] 

clf.fit(xtr, ytr)
yhts = clf.predict(xts)
yhts = pd.DataFrame(yhts)

# =============================================================================
# Plot graph
# =============================================================================


n_point = 1000 - 1

# show only the last column of output, the 24th column.
plt.scatter(yts.loc[tr_nrows:tr_nrows+n_point,123], yhts.loc[0:n_point,23])
t = np.arange(0, 1, 0.1)
plt.figure(1)
plt.plot(t, t, color='green')

# =============================================================================
# Mean squared error
# =============================================================================

rms = []
gamma_range = []
    
for count in range(0, 1000):
    gamma = count * 0.1
    clf = Ridge(alpha = gamma)
    clf.fit(x, y)
    Ridge(alpha=gamma, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)
    yths = clf.predict(x)
    gamma_range.append(gamma)
    rms.append(sqrt(mean_squared_error(yts, yhts)))
    
plt.figure(1)
plt.plot(gamma_range, rms)
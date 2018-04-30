

from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import math
from sklearn.metrics import mean_squared_error

# =============================================================================
# read data from csv
# =============================================================================
#num_data = pd.read_csv('../data/gencsv_ep100_vec100_Spre.csv'
#                        ,header=None,sep=',',error_bad_lines=False,encoding='utf-8')

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
# alpha is gamma, the coefficient of panelty term
clf = Ridge(alpha=1.0) 

clf.fit(x, y) 
Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
clf.coef_

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
y_pred = yhts.loc[0:n_point,23]
y_test = yts.loc[tr_nrows:tr_nrows+n_point,123]

# show only the last output column, the 24th column.
plt.scatter(y_test, y_pred)
t = np.arange(0, 1, 0.1)
plt.figure(1)
plt.plot(t, t, color='green')

# =============================================================================
# Evaluation
# =============================================================================
#RMSE
rmse = math.sqrt(mean_squared_error(y_pred,y_test))

# =============================================================================
# rmse against gamma
# =============================================================================



# =============================================================================
# number of non-zero coefficient against gamma
# =============================================================================











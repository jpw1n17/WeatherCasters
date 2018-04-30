

from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import math
from sklearn.metrics import mean_squared_error

# =============================================================================
# read data from csv
# =============================================================================
train_data = pd.read_csv('../data/gencsv_ep100_vec100_Spre.csv'
                        ,header=None,sep=',',error_bad_lines=False,encoding='utf-8')

test_data = pd.read_csv('../data/gencsv_ep100_vec100_Spre.csv'
                        ,header=None,sep=',',error_bad_lines=False,encoding='utf-8')

## =============================================================================
## put data into variables
## =============================================================================
n_output = 24
n_datacol = len(train_data.columns)

n_input = n_datacol-n_output

ytr = train_data.loc[:,n_input:n_input+n_output-1]
xtr = train_data.loc[:,0:n_input-1]

xts = test_data

# =============================================================================
# create model
# =============================================================================
# alpha is gamma, the coefficient of panelty term
clf = Ridge(alpha=1.0) 

Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
#clf.coef_

# =============================================================================
# train model
# =============================================================================

clf.fit(xtr, ytr)                         
yhts = clf.predict(xts)
#yhts = pd.DataFrame(yhts)

# =============================================================================
# 
# =============================================================================










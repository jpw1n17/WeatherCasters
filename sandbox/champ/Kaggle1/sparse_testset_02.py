

from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import math
from sklearn.metrics import mean_squared_error

# =============================================================================
# read data from csv
# =============================================================================
#train_data = pd.read_csv('../data/gencsv_ep100_vec100_Spre.csv'
#                        ,header=None,sep=',',error_bad_lines=False,encoding='utf-8')
#
#test_data = pd.read_csv('../data/gencsv_ep100_vec100_Spre_test_vec.txt'
#                        ,header=None,sep='\t',error_bad_lines=False,encoding='utf-8')

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
yhts = pd.DataFrame(yhts)

# =============================================================================
# Get the id of the test data
# =============================================================================

#test_orig = pd.read_csv('../data/test.csv'
#                         ,sep=',',error_bad_lines=False,encoding='utf-8')

test_row_id = test_orig["id"].astype(int)

# =============================================================================
# Add the id of the test data
# =============================================================================

yhts = yhts.assign(id=pd.Series(test_row_id))

# list the names of the columns
yhts_cols = yhts.columns.tolist()

# change the order of the columns (the last one becomes first = id is shifted to the first column)
yhts_cols = yhts_cols[-1:] +  yhts_cols[:-1]

# the dataframe is reordered
yhts = yhts[yhts_cols]


# =============================================================================
# write into the csv file
# =============================================================================

print( 'number of records : ' + str(len(yhts)) )

hdr ="id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15"

yhts_fmt = '%d'

for i in range(24):
    yhts_fmt = yhts_fmt + ',%.8f'

np.savetxt('gencsv_ep100_vec100_Spre_predict.csv',yhts,fmt=yhts_fmt,delimiter=',',comments='',header=hdr)

# =============================================================================
# 
# =============================================================================










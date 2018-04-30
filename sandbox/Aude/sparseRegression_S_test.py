from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir('C:/Users/audeb/Southampton Uni/Sem2/Data Mining/Test set')


# =============================================================================
# read data from csv
# =============================================================================
S_train = pd.read_csv('Train data/gencsv_ep100_vec100_Spre.csv'
                        ,header=None,sep=',',error_bad_lines=False,encoding='utf-8')

## =============================================================================
## put data into variables
## =============================================================================
n_output = 24
n_datacol = len(S_train.columns)

n_input = n_datacol-n_output

y = S_train.loc[:,n_input:n_input+n_output-1]
x = S_train.loc[:,0:n_input-1]

# =============================================================================
# create model
# =============================================================================
gamma = 0.2
clf = Ridge(alpha = gamma)
#clf.fit(x, y)
Ridge(alpha=gamma, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)

## =============================================================================
## train model
## =============================================================================
#
#all_nrows = len(x)
#tr_nrows = round(all_nrows*0.8) - 1
#
#ytr = y[0:tr_nrows]
#xtr = x[0:tr_nrows]
#
#yts = y[tr_nrows:all_nrows]
#xts = x[tr_nrows:all_nrows] 
#
#clf.fit(xtr, ytr)
#yhts = clf.predict(xts)
#yhts = pd.DataFrame(yhts)
#
## =============================================================================
## Plot graph
## =============================================================================
#
#
#n_point = 1000 - 1
#
## show only the last column of output, the 24th column.
#plt.scatter(yts.loc[tr_nrows:tr_nrows+n_point,123], yhts.loc[0:n_point,23])
#t = np.arange(0, 1, 0.1)
#plt.figure(1)
#plt.plot(t, t, color='green')
#
## =============================================================================
# Mean squared error
# =============================================================================

#rms = []
#gamma_range = []
#    
#for count in range(0, 1000):
#    gamma = count * 0.01
#    clf = Ridge(alpha = gamma)
#    clf.fit(x, y)
#    Ridge(alpha=gamma, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)
#    yths = clf.predict(x)
#    gamma_range.append(gamma)
#    rms.append(sqrt(mean_squared_error(yts, yhts)))
#    
#plt.figure(1)
#plt.plot(gamma_range, rms,'o')

# =============================================================================
# Read the test data
# =============================================================================

Spre_Test = pd.read_csv('gencsv_ep100_vec100_Spre_test_vec.csv'
                        ,header=None,sep='\t',error_bad_lines=False,encoding='utf-8')

Test = pd.read_csv('test.csv'
                        ,sep=',',error_bad_lines=False,encoding='utf-8')

## =============================================================================
## put data into variables
## =============================================================================

xts = Spre_Test
test_row_id = Test["id"].astype(int)

# =============================================================================
# train model
# =============================================================================

ytr = y
xtr = x

clf.fit(xtr, ytr)
yhts = clf.predict(xts)


yhts = pd.DataFrame(yhts)

yhts = yhts.assign(id=pd.Series(test_row_id))

# list the names of the columns
cols = yhts.columns.tolist()

# change the order of the columns (the last one becomes first = id is shifted to the first column)
cols = cols[-1:] + cols[:-1]

# the dataframe is reordered
yhts = yhts[cols]

print( 'number of records : ' + str(len(yhts)) )

#np.savetxt('gencsv_ep100_vec100_Spre_predict.csv',yhts,fmt='%.8f',delimiter=',', comments='', header ="id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15")

np.savetxt('gencsv_ep100_vec100_Spre_predict.csv',yhts,fmt=' '.join(['%i'] + ['%.8f']*24),delimiter=',', comments='', header ="id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15")


# =============================================================================
# 
# =============================================================================



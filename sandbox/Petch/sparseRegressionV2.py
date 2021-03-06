import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
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

Y = num_data.loc[:,n_input:n_input+n_output-1]
X = num_data.loc[:,0:n_input-1]

# =============================================================================
# k-fold cross validation
# =============================================================================
kf = KFold(n_splits=5)
kf.get_n_splits(X)

x = X.values
y = Y.values

n_alphas = 4 
alphas = np.logspace(-12, 2, n_alphas)
rms_errs_all = []
alphas_all = []
for a in alphas:
    rms_errs = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
        ridge.fit(x_train, y_train)
        ridge.coef_
        
        y_predicted = ridge.predict(x_test)
        rms = sqrt(mean_squared_error(y_test, y_predicted))
        rms_errs.append(rms)
    rms_errs_all.append(rms_errs)
    alphas_all.append(a)
    
colors = ['pink', 'lightblue', 'lightgreen', 'yellow']
box = plt.boxplot(rms_errs_all, labels = alphas_all)

# =============================================================================
# # #############################################################################
# # Compute paths
# 
# n_alphas = 100 
# alphas = np.logspace(-10, -2, n_alphas)
# 
# coefs = []
# for a in alphas:
#     ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
#     ridge.fit(x, y)
#     coefs.append(ridge.coef_)
# 
# coefs2 = []
# for i in range(0, n_alphas):
#     coef_temp = coefs[i][0]
#     coefs2.append(coef_temp)
#     
# 
# # #############################################################################
# # Display results
#     
# ax = plt.gca()
# ax.plot(alphas, coefs2)
# ax.set_xscale('log')
# ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
# plt.xlabel('alpha')
# plt.ylabel('weights')
# plt.title('Ridge coefficients as a function of the regularization')
# plt.axis('tight')
# plt.show()
# =============================================================================
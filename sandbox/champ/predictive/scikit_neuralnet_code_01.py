
from sklearn.neural_network import MLPRegressor
#from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# read data from csv
# =============================================================================
num_data = pd.read_csv('gencsv_ep100_vec100_Spre.csv'
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
mlp = MLPRegressor(activation='logistic',solver='sgd', alpha=1e-5,
                     hidden_layer_sizes=(100,), random_state=1,
                     early_stopping=False,momentum=0.9,
                     batch_size='auto')

#mlp = MLPClassifier(activation='logistic',solver='sgd', alpha=1e-5,
#                     hidden_layer_sizes=(100,), random_state=1,
#                     early_stopping=False,momentum=0.9,
#                     batch_size='auto')
# =============================================================================
# train model
# =============================================================================

all_nrows = len(x)
tr_nrows = round(all_nrows/10) - 1

ytr = y[0:tr_nrows]
xtr = x[0:tr_nrows]

yts = y[tr_nrows:all_nrows]
xts = x[tr_nrows:all_nrows] 

mlp.fit(xtr, ytr)                         
yhts = mlp.predict(xts)
yhts = pd.DataFrame(yhts)

# =============================================================================
# Plot graph
# =============================================================================

n_point = 1000 - 1
plt.scatter(yts.loc[7794:7794+n_point,123], yhts.loc[0:n_point,23])
t = np.arange(0, 1, 0.1)
plt.figure(1)
plt.plot(t, t, color='green')

# =============================================================================
# 
# =============================================================================











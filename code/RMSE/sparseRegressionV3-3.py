import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import os

#os.chdir('D:\Learning\Semester 2\DM\CW 1')
os.chdir('D:\Champ\AnacondaProjects\WeatherCasters\Local\sandbox\champ')

# =============================================================================
# read data from csv
# =============================================================================
# =============================================================================
# num_data = pd.read_csv('Data\gencsv_ep100_vec100_Kpre_test_vec.txt'
#                         ,header=None,sep=',',error_bad_lines=False,encoding='utf-8')
# =============================================================================

# =============================================================================
# read test vector dataset
# =============================================================================

#vec_testK = pd.read_csv('Data\gencsv_ep100_vec100_Kpre_train_vec.txt'
#                       ,header=None,sep='\t',error_bad_lines=False,encoding='utf-8')
#vec_testS = pd.read_csv('Data\gencsv_ep100_vec100_Spre_test_vec.txt'
#                       ,header=None,sep='\t',error_bad_lines=False,encoding='utf-8')
#vec_testW = pd.read_csv('Data\gencsv_ep100_vec100_Wpre_test_vec.txt'
#                       ,header=None,sep='\t',error_bad_lines=False,encoding='utf-8')

# =============================================================================
# read training vector dataset
# =============================================================================

num_dataK = pd.read_csv('Data\gencsv_ep100_vec100_Kpre.txt'
                       ,header=None,sep='\t',error_bad_lines=False,encoding='utf-8')
num_dataS = pd.read_csv('Data\gencsv_ep100_vec100_Spre.csv'
                       ,header=None,sep=',',error_bad_lines=False,encoding='utf-8')
num_dataW = pd.read_csv('Data\gencsv_ep100_vec100_Wpre.txt'
                       ,header=None,sep='\t',error_bad_lines=False,encoding='utf-8')

# =============================================================================
# cut data out
# =============================================================================
#num_dataK = num_dataK[0: 500]
#num_dataS = num_dataS[0: 500]
#num_dataW = num_dataW[0: 500]

# =============================================================================
# function to create and train model 
# =============================================================================
   
def create_model(x_train, y_train,gamma):
    ridge = linear_model.Ridge(alpha=gamma, fit_intercept=False)
    ridge.fit(x_train, y_train)
    ridge.coef_
    return ridge
    
## =============================================================================
## put data into variables
## =============================================================================

def split_input_output(num_data):
    n_output = 24
    n_datacol = len(num_data.columns)
    
    n_input = n_datacol-n_output
    
    Y = num_data.loc[:, n_input:n_input+n_output-1]
    X = num_data.loc[:, 0:n_input-1]
    
    return X,Y

def to_array(X,Y):
    x = X.values
    y = Y.values
    return x,y

# =============================================================================
# split train set and test set for cross-validation
# =============================================================================

def split_train_test(x,y,train_index,test_index):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    return x_train, x_test,y_train, y_test


# =============================================================================
# Cross validation function
# =============================================================================

def create_model_kfolds(k_folds,x,y,gamma):
    kfolds_model = []
    kfolds_x_train = []
    kfolds_x_test = []
    kfolds_y_train = []
    kfolds_y_test = []
    
    kf = KFold(n_splits=k_folds)
    kf.get_n_splits(x)
    for train_index, test_index in kf.split(x):
        x_train, x_test,y_train, y_test = split_train_test(x,y,train_index,test_index)
        model = create_model(x_train,y_train,gamma)
            
        kfolds_model.append(model)
        kfolds_x_train.append(x_train)
        kfolds_x_test.append(x_test)
        kfolds_y_train.append(y_train)
        kfolds_y_test.append(y_test)
    return kfolds_model, kfolds_x_train, kfolds_x_test, kfolds_y_train, kfolds_y_test

# =============================================================================
# Main program
# =============================================================================

n_output = 24    
n_datacol = len(num_dataS.columns)
n_input = n_datacol-n_output

# split input data and output data
x_S, y_S_1 = split_input_output(num_dataS)
x_W, y_W_1 = split_input_output(num_dataW)
x_K, y_K_1 = split_input_output(num_dataK)

# select some output columns
y_S = y_S_1.loc[:, n_input : n_input+4 ]
y_W = y_W_1.loc[:, n_input+5 : n_input+8]
y_K = y_K_1.loc[:, n_input+9 : n_input+23 ]

# transform to array
x_S,y_S = to_array(x_S,y_S)
x_W,y_W = to_array(x_W,y_W )
x_K,y_K = to_array(x_K,y_K)

# number of fold for cross-validation
num_folds = 5

# set gamma/alpha value
#n_alphas = 12
#alphas = np.logspace(-6, 1, n_alphas)
#alphas = [0.2, 0.5, 1, 2, 5]
alphas = [0.2]


rms_errs_all = []
alphas_all = []
rms_output_all = []

for gamma in alphas: 
    kfolds_model_S, kfolds_x_train_S, kfolds_x_test_S, kfolds_y_train_S, kfolds_y_test_S = create_model_kfolds(num_folds,x_S,y_S,gamma)
    kfolds_model_W, kfolds_x_train_W, kfolds_x_test_W, kfolds_y_train_W, kfolds_y_test_W = create_model_kfolds(num_folds,x_W,y_W,gamma) 
    kfolds_model_K, kfolds_x_train_K, kfolds_x_test_K, kfolds_y_train_K, kfolds_y_test_K = create_model_kfolds(num_folds,x_K,y_K,gamma)

    #print('S model')
    kfolds_yh_test_S = []
    for i in range(num_folds):
        y_predicted = kfolds_model_S[i].predict(kfolds_x_test_S[i])
        kfolds_yh_test_S.append(y_predicted)
        
    #print('W model')
    kfolds_yh_test_W = []
    for i in range(num_folds):
        y_predicted = kfolds_model_W[i].predict(kfolds_x_test_W[i])
        kfolds_yh_test_W.append(y_predicted)
    
    #print('K model')
    kfolds_yh_test_K = []
    for i in range(num_folds):
        y_predicted = kfolds_model_K[i].predict(kfolds_x_test_K[i])
        kfolds_yh_test_K.append(y_predicted)
     
    # Calculate RMSE    
    rms_folds = []
    rms_output_folds = []
    for i in range(num_folds):
        yh_test = np.append(kfolds_yh_test_S[i], kfolds_yh_test_W[i], axis=1)
        yh_test = np.append(yh_test, kfolds_yh_test_K[i], axis=1)
    
        y_test = np.append(kfolds_y_test_S[i], kfolds_y_test_W[i], axis=1)
        y_test = np.append(y_test, kfolds_y_test_K[i], axis=1)
        
        rms = sqrt(mean_squared_error(y_test, yh_test))
        
        rms_output = []
        for n_op in range(n_output):
            rms_each_output = sqrt(mean_squared_error(y_test[:,n_op], yh_test[:,n_op]))
            rms_output.append(rms_each_output)
        
        rms_folds.append(rms)
        rms_output_folds.append(rms_output)
    rms_errs_all.append(rms_folds)
    alphas_all.append(gamma)
    rms_output_all.append(rms_output_folds)

## =============================================================================
## plot graph 1    
## =============================================================================
        
box = plt.boxplot(rms_errs_all, labels = alphas_all)        

## =============================================================================
## plot graph 2    
## =============================================================================

# create a list of the benchmark value to plot line graph        
benchmark_val = 0.31957
baseval_lst = [benchmark_val for i in range(len(alphas_all))]


# bigger font size
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

plt.rc('font', **font)


# plot figure
plt.figure(2)
fig, ax = plt.subplots(figsize=(10, 6))

bp = ax.boxplot(rms_errs_all, labels = alphas_all, 
                #positions = alphas_all,
                  notch=0, vert=1, whis=1.5)

#ax.set_xlim(0, max(alphas_all)+1)
#ax.set_xticks(alphas_all)
ax.set_xticklabels(alphas_all) 

plt.setp(bp['boxes'], color='blue')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='black', marker='.')
plt.setp(bp['medians'], color='orange')

# plot the benchmark value line graph
#lngrp = plt.plot(alphas_all, baseval_lst , 'b-o', label='base-line') 


# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

# Hide these grid behind plot objects
ax.set_axisbelow(True)
ax.set_title('Sparse Regression RMSE over gamma')
ax.set_xlabel('Gamma')
ax.set_ylabel('RMSE')

plt.legend()
plt.show()

# =============================================================================
# Calculate the Average RMS Error of each output
# =============================================================================

rms_output_col = np.zeros(n_output, dtype=np.float)
rms_output_avg = np.zeros(n_output, dtype=np.float)

for rms_output_row in rms_output_folds:       
    for n_opt in range(n_output):
        rms_output_col[n_opt] = rms_output_col[n_opt] + rms_output_row[n_opt]
        
    for n_opt in range(n_output):
        rms_output_avg[n_opt] = rms_output_col[n_opt]/num_folds


# =============================================================================
# Get the result of the Average RMS Error of each output
# =============================================================================

result = rms_output_avg

print(result)
avg_result = np.mean(result)
print("AVG of RMSE: "+ str(avg_result))


# =============================================================================
# Plot the bar chart of the RMSE of every output
# =============================================================================

N = len(result)
x = range(1,N+1)

fig, ax = plt.subplots(figsize=(10, 6))
ax = plt.subplot(111)
ax.set_title('RMSE from Sparse Regression')
ax.set_xlabel('output index')
ax.set_xticks(x)
ax.set_ylabel('RMSE')
ax.bar(x, result)
ax.axhline(y=avg_result, color='r', linestyle='-')


# =============================================================================



# =============================================================================
# 
# =============================================================================

# =============================================================================
# Previous main program
# =============================================================================


# =============================================================================
# k-fold cross validation
# =============================================================================
#kf = KFold(n_splits=10)
#kf.get_n_splits(X)
#
#
#n_alphas = 6
#alphas = np.logspace(-4, 1, n_alphas)
#rms_errs_all = []
#alphas_all = []
#for a in alphas:
#    rms_errs = []
#    for train_index, test_index in kf.split(x):
#        x_train, x_test = x[train_index], x[test_index]
#        y_train, y_test = y[train_index], y[test_index]
#        
#        create_model(x_train,y_train,a)
#        
#        y_predicted = ridge.predict(x_test)
#        rms = sqrt(mean_squared_error(y_test, y_predicted))
#        rms_errs.append(rms)
#    rms_errs_all.append(rms_errs)
#    alphas_all.append(a)
#
## =============================================================================
## plot graph    
## =============================================================================
#colors = ['pink', 'lightblue', 'lightgreen', 'yellow']
#box = plt.boxplot(rms_errs_all, labels = alphas_all)
#
#


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
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import os
import lightgbm as lgb
os.chdir('G:\work\WeatherCasters')

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

num_dataK = pd.read_csv('code\gencsv\gencsv_ep100_vec100_Kpre.txt'
                       ,header=None,sep='\t',error_bad_lines=False,encoding='utf-8')
num_dataS = pd.read_csv('code\gencsv\gencsv_ep100_vec100_Spre.csv'
                       ,header=None,sep=',',error_bad_lines=False,encoding='utf-8')
num_dataW = pd.read_csv('code\gencsv\gencsv_ep100_vec100_Wpre.txt'
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
   
def create_model(x_train, y_train, params):
    d_train = lgb.Dataset(x_train, y_train.flatten())
    params = {}
    params['learning_rate'] = 0.05
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'regression_l2'
    params['metric'] = 'l2_root'
    params['sub_feature'] = 0.5
    params['num_leaves'] = 25
    params['min_data'] = 50
    params['max_depth'] = 10
    params['is_unbalance'] = True
    params['num_iterations'] = 1000
    
    #======Training model
    model = lgb.train(params, d_train, 100)
    
    return model

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
# k | model[24] | x_test | y_test[24]
def create_model_kfolds(k_folds,x,y,params):
    kfolds_model = []
    kfolds_x_train = []
    kfolds_x_test = []
    kfolds_y_train = []
    kfolds_y_test = []
    
    kf = KFold(n_splits=k_folds)
    kf.get_n_splits(x)
    for train_index, test_index in kf.split(x):
        x_train, x_test,y_train, y_test = split_train_test(x,y,train_index,test_index)
        
        model = create_model(x_train,y_train,params)
            
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
x_S, y_W_1 = split_input_output(num_dataS)
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
num_folds = 10

# set gamma/alpha value
#n_alphas = 12
#alphas = np.logspace(-6, 1, n_alphas)
#alphas = [0.2, 0.5, 1, 2, 5]

rms_errs_all = []
alphas_all = []

kfolds_model_S, kfolds_x_train_S, kfolds_x_test_S, kfolds_y_train_S, kfolds_y_test_S = [None] * y_S.shape[1],[None] * y_S.shape[1],[None] * y_S.shape[1],[None] * y_S.shape[1],[None] * y_S.shape[1]
kfolds_model_W, kfolds_x_train_W, kfolds_x_test_W, kfolds_y_train_W, kfolds_y_test_W = [None] * y_W.shape[1],[None] * y_W.shape[1],[None] * y_W.shape[1],[None] * y_W.shape[1],[None] * y_W.shape[1]
kfolds_model_K, kfolds_x_train_K, kfolds_x_test_K, kfolds_y_train_K, kfolds_y_test_K = [None] * y_K.shape[1],[None] * y_K.shape[1],[None] * y_K.shape[1],[None] * y_K.shape[1],[None] * y_K.shape[1]

params1 = {}
params1['learning_rate'] = 0.05
params1['boosting_type'] = 'gbdt'
params1['objective'] = 'regression_l2'
params1['metric'] = 'l2_root'
params1['sub_feature'] = 0.5
params1['num_leaves'] = 25
params1['min_data'] = 50
params1['max_depth'] = 10
params1['is_unbalance'] = True
params1['num_iterations'] = 1000
    
params_list = [params1]

#test
num_folds = 2
for s in range(y_S.shape[1]):
        kfolds_model_S[s],kfolds_x_train_S[s],kfolds_x_test_S[s],kfolds_y_train_S[s],kfolds_y_test_S[s] = create_model_kfolds(num_folds,x_S,y_S[:,s],params1)

S_predicted = []
for s in range(y_S.shape[1]):
        model = kfolds_model_S[s]
        x_test = kfolds_x_test_S[s]
        folds_predicted = []
        #predict each fold under each S model (there are 5 models for S)
        for k in range(num_folds):
            y_predicted = model[k].predict(x_test[k])
            folds_predicted.append(y_predicted)
            
        S_predicted.append(folds_predicted)

##
        
#Structure kfolds_model_S => Model_S[S1-S5][K] 
for params in params_list: 
    # create 24 x k model for each fold
    for s in range(y_S.shape[1]):
        kfolds_model_S[s],kfolds_x_train_S[s],kfolds_x_test_S[s],kfolds_y_train_S[s],kfolds_y_test_S[s] = create_model_kfolds(num_folds,x_S,y_S[:,s],params)
        
    for w in range(y_W.shape[1]):
        kfolds_model_W[w], kfolds_x_train_W[w], kfolds_x_test_W[w], kfolds_y_train_W[w], kfolds_y_test_W[w] = create_model_kfolds(num_folds,x_W,y_W[:,w],params) 
    
    for k in range(y_K.shape[1]):
        kfolds_model_K[k], kfolds_x_train_K[k], kfolds_x_test_K[k], kfolds_y_train_K[k], kfolds_y_test_K[k] = create_model_kfolds(num_folds,x_K,y_K[:,k],params)

#    kfolds_yh_test_S = []
    # Predict each model in S category
    S_predicted = []
    for i in range(y_S.shape[1]):
        model = kfolds_model_S[i]
        x_test = kfolds_x_test_S[i]
        folds_predicted = []
        #predict each fold under each S model (there are 5 models for S)
        for k in range(num_folds):
            y_predicted = model[k].predict(x_test[k])
            folds_predicted.append(y_predicted)
        S_predicted.append(y_predicted)
        
        
    # Predict each model in W category
    W_predicted = []
    for i in range(y_W.shape[1]):
        model = kfolds_model_W[i]
        x_test = kfolds_x_test_W[i]
        folds_predicted = []
        #predict each fold under each W model 
        for k in range(num_folds):
            y_predicted = model[k].predict(x_test[k])
            folds_predicted.append(y_predicted)
        W_predicted.append(y_predicted)
      
    # Predict each model in K category
    K_predicted = []
    for i in range(y_K.shape[1]):
        model = kfolds_model_K[i]
        x_test = kfolds_x_test_K[i]
        folds_predicted = []
        #predict each fold under each W model 
        for k in range(num_folds):
            y_predicted = model[k].predict(x_test[k])
            folds_predicted.append(y_predicted)
        K_predicted.append(y_predicted)
        
    # Calculate RMSE    
    rms_s = []
    rms_w = []
    rms_k = []
    
    for i in range(y_S.shape[1]):
        for k in range(num_folds):
            rms = sqrt(mean_squared_error(kfolds_x_test_S[i],S_predicted[i])) 
            rms_s.append(rms)
            
    for i in range(y_W.shape[1]):
        for k in range(num_folds):
            rms = sqrt(mean_squared_error(kfolds_x_test_W[i],W_predicted[i])) 
            rms_w.append(rms)
            
    for i in range(y_K.shape[1]):
        for k in range(num_folds):
            rms = sqrt(mean_squared_error(kfolds_x_test_K[i],K_predicted[i])) 
            rms_k.append(rms)
    
    # Calculate all rmse
    # 
    rms_all = 0     
            
    rms_folds = []
    
    for i in range(num_folds):
        yh_test = np.append(kfolds_yh_test_S[i], kfolds_yh_test_W[i], axis=1)
        yh_test = np.append(yh_test, kfolds_yh_test_K[i], axis=1)
    
        y_test = np.append(kfolds_y_test_S[i], kfolds_y_test_W[i], axis=1)
        y_test = np.append(y_test, kfolds_y_test_K[i], axis=1)
        
        rms = sqrt(mean_squared_error(y_test, yh_test))
        rms_folds.append(rms)
    rms_errs_all.append(rms_folds)
    alphas_all.append(gamma)

    # Count number of non-zero coefficient
    for i in range(num_folds):
        onefold_model = np.append(kfolds_model_S[i], kfolds_model_W[i], axis=1)
        onefold_model = np.append(onefold_model, kfolds_model_K[i], axis=1)
        
        


## =============================================================================
## plot graph    
## =============================================================================
        
box = plt.boxplot(rms_errs_all, labels = alphas_all)        
        

# =============================================================================
# Previou main program
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
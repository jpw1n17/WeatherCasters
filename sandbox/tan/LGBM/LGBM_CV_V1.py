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
import time
starttime = time.time()
os.chdir('G:\work\WeatherCasters')

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def load_object(path):
    with open(path, 'rb') as input:
        obj = pickle.load(input)
    return obj   
def genS4_error():
    from sklearn.cross_validation import train_test_split
    X = x_S
    y = y_S[:,3]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    
    #======Setup parameters
    d_train = lgb.Dataset(x_train, y_train.flatten())
    
    params = {}
    params['learning_rate'] = 0.07
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'regression_l2'
    params['metric'] = 'l2_root'
    params['sub_feature'] = 0.5
    params['num_leaves'] = 120
    params['min_data'] = 50
    params['max_depth'] = 10
    params['is_unbalance'] = True
    params['num_iterations'] = 1000
    params['reg_lambda'] = 0.02
    
    
    #======Training model
    clf = lgb.train(params, d_train, 100)
    
    #======Prediction
    y_pred=clf.predict(x_test)
    y_test.shape
    y_pred.shape
    error = abs(y_pred - y_test)
    error
    
    yall_pred = clf.predict(X)
    yall_pred.shape
    y.shape
    error = abs(yall_pred - y)
    max_error = max(error)
    max_error 
    list(error).index(max(error))
    
    
    output = pd.DataFrame()
    output['s4_error'] = error
    hdr ="s4_abs_error"
    #output_format = '%d'
    
    output_format = '%.8f'
    
    np.savetxt('s4_abs_error.csv',output,fmt=output_format,delimiter=',',comments='',header=hdr)
    ##
      
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
##for testing
#num_dataK = num_dataK.sample(frac=0.1)
#num_dataS = num_dataS.sample(frac=0.1)
#num_dataW = num_dataW.sample(frac=0.1)
####

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
    round=1
    for train_index, test_index in kf.split(x):
        print('---Training K='+str(round)+' model...')
        x_train, x_test,y_train, y_test = split_train_test(x,y,train_index,test_index)
        
        model = create_model(x_train,y_train,params)
            
        kfolds_model.append(model)
        kfolds_x_train.append(x_train)
        kfolds_x_test.append(x_test)
        kfolds_y_train.append(y_train)
        kfolds_y_test.append(y_test)
        round+=1
    return kfolds_model, kfolds_x_train, kfolds_x_test, kfolds_y_train, kfolds_y_test

# =============================================================================
# Main program
# =============================================================================

n_output = 24    
n_datacol = len(num_dataS.columns)
n_input = n_datacol-n_output

# split input data and output data
x_S, y_S = split_input_output(num_dataS)
x_W, y_W = split_input_output(num_dataW)
x_K, y_K = split_input_output(num_dataK)

# select some output columns
y_S = y_S.loc[:, n_input : n_input+4 ]
y_W = y_W.loc[:, n_input+5 : n_input+8]
y_K = y_K.loc[:, n_input+9 : n_input+23 ]


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


kfolds_model_S, kfolds_x_train_S, kfolds_x_test_S, kfolds_y_train_S, kfolds_y_test_S = [None] * y_S.shape[1],[None] * y_S.shape[1],[None] * y_S.shape[1],[None] * y_S.shape[1],[None] * y_S.shape[1]
kfolds_model_W, kfolds_x_train_W, kfolds_x_test_W, kfolds_y_train_W, kfolds_y_test_W = [None] * y_W.shape[1],[None] * y_W.shape[1],[None] * y_W.shape[1],[None] * y_W.shape[1],[None] * y_W.shape[1]
kfolds_model_K, kfolds_x_train_K, kfolds_x_test_K, kfolds_y_train_K, kfolds_y_test_K = [None] * y_K.shape[1],[None] * y_K.shape[1],[None] * y_K.shape[1],[None] * y_K.shape[1],[None] * y_K.shape[1]

## =============================================================================
## Setup testing params
## =============================================================================
nlv = [20,25,30,35,40]
params1 = {}
params1['learning_rate'] = 0.03
params1['boosting_type'] = 'gbdt'
params1['objective'] = 'regression_l2'
params1['metric'] = 'l2_root'
params1['sub_feature'] = 0.5
params1['num_leaves'] = nlv[0]
params1['min_data'] = 50
params1['max_depth'] = 10
params1['is_unbalance'] = True
params1['num_iterations'] = 1000
params1['reg_lambda'] = 0.02

params2 = {}
params2['learning_rate'] = 0.03
params2['boosting_type'] = 'gbdt'
params2['objective'] = 'regression_l2'
params2['metric'] = 'l2_root'
params2['sub_feature'] = 0.5
params2['num_leaves'] = nlv[1]
params2['min_data'] = 50
params2['max_depth'] = 10
params2['is_unbalance'] = True
params2['num_iterations'] = 1000
params2['reg_lambda'] = 0.02

params3 = {}
params3['learning_rate'] = 0.03
params3['boosting_type'] = 'gbdt'
params3['objective'] = 'regression_l2'
params3['metric'] = 'l2_root'
params3['sub_feature'] = 0.5
params3['num_leaves'] = nlv[2]
params3['min_data'] = 50
params3['max_depth'] = 10
params3['is_unbalance'] = True
params3['num_iterations'] = 1000
params3['reg_lambda'] = 0.02

params4 = {}
params4['learning_rate'] = 0.03
params4['boosting_type'] = 'gbdt'
params4['objective'] = 'regression_l2'
params4['metric'] = 'l2_root'
params4['sub_feature'] = 0.5
params4['num_leaves'] = nlv[3]
params4['min_data'] = 50
params4['max_depth'] = 10
params4['is_unbalance'] = True
params4['num_iterations'] = 1000
params4['reg_lambda'] = 0.02

params5 = {}
params5['learning_rate'] = 0.03
params5['boosting_type'] = 'gbdt'
params5['objective'] = 'regression_l2'
params5['metric'] = 'l2_root'
params5['sub_feature'] = 0.5
params5['num_leaves'] = nlv[4]
params5['min_data'] = 50
params5['max_depth'] = 10
params5['is_unbalance'] = True
params5['num_iterations'] = 1000
params5['reg_lambda'] = 0.02

params_list = [params1,params2,params3,params4,params5]

##test
#genS4_error()


## =============================================================================
## Train and test model to compare performance btw each configurations     
## =============================================================================
rms_allParams_Ncategory_Kfolds = []
rms_all_params = []

for idx,params in enumerate(params_list):
    
    # create 24 x k model for each fold
    for s in range(y_S.shape[1]):
        print(str(idx)+'_Training S'+ str(s) +' model...')
        kfolds_model_S[s],kfolds_x_train_S[s],kfolds_x_test_S[s],kfolds_y_train_S[s],kfolds_y_test_S[s] = create_model_kfolds(num_folds,x_S,y_S[:,s],params)
        
    for w in range(y_W.shape[1]):
        print(str(idx)+'_Training W'+ str(w) +' model...')
        kfolds_model_W[w], kfolds_x_train_W[w], kfolds_x_test_W[w], kfolds_y_train_W[w], kfolds_y_test_W[w] = create_model_kfolds(num_folds,x_W,y_W[:,w],params) 
    
    for k in range(y_K.shape[1]):
        print(str(idx)+'_Training K'+ str(k) +' model...')
        kfolds_model_K[k], kfolds_x_train_K[k], kfolds_x_test_K[k], kfolds_y_train_K[k], kfolds_y_test_K[k] = create_model_kfolds(num_folds,x_K,y_K[:,k],params)

    # Predict each model in S category
    S_predicted = []
    
    for i in range(y_S.shape[1]):
        print(str(idx)+'_Testing S'+ str(i) +' model...')
        model = kfolds_model_S[i]
        x_test = kfolds_x_test_S[i]
        folds_predicted = []
        #predict each fold under each S model (there are 5 models for S)
        for k in range(num_folds):
            y_predicted = model[k].predict(x_test[k])
            folds_predicted.append(y_predicted)
        S_predicted.append(folds_predicted)
        
        
    # Predict each model in W category
    W_predicted = []
    for i in range(y_W.shape[1]):
        print(str(idx)+'_Testing W'+ str(i) +' model...')
        model = kfolds_model_W[i]
        x_test = kfolds_x_test_W[i]
        folds_predicted = []
        #predict each fold under each W model 
        for k in range(num_folds):
            y_predicted = model[k].predict(x_test[k])
            folds_predicted.append(y_predicted)
        W_predicted.append(folds_predicted)
      
    # Predict each model in K category
    K_predicted = []
    for i in range(y_K.shape[1]):
        print(str(idx)+'_Testing K'+ str(i) +' model...')
        model = kfolds_model_K[i]
        x_test = kfolds_x_test_K[i]
        folds_predicted = []
        #predict each fold under each W model 
        for k in range(num_folds):
            y_predicted = model[k].predict(x_test[k])
            folds_predicted.append(y_predicted)
        K_predicted.append(folds_predicted)
        
    # Calculate RMSE    
    rms_s = []    
    rms_w = []
    rms_k = []
    
    for i in range(y_S.shape[1]):
        print(str(idx)+'_Calculate RMSE for S'+ str(i) +' model...')
        folds_rms = []
        for k in range(num_folds):
            rms = sqrt(mean_squared_error(kfolds_y_test_S[i][k],S_predicted[i][k])) 
            folds_rms.append(rms)
        #assign all k-fold result to each model
        rms_s.append(folds_rms)        
        
    for i in range(y_W.shape[1]):
        print(str(idx)+'_Calculate RMSE for W'+ str(i) +' model...')
        folds_rms = []
        for k in range(num_folds):
            rms = sqrt(mean_squared_error(kfolds_y_test_W[i][k],W_predicted[i][k])) 
            folds_rms.append(rms)  
        #assign all k-fold result to each model
        rms_w.append(folds_rms)      
        
    for i in range(y_K.shape[1]):
        print(str(idx)+'_Calculate RMSE for K'+ str(i) +' model...')
        folds_rms = []
        for k in range(num_folds):
            rms = sqrt(mean_squared_error(kfolds_y_test_K[i][k],K_predicted[i][k])) 
            folds_rms.append(rms)    
        #assign all k-fold result to each model
        rms_k.append(folds_rms)        
     
    print('Combining result...')   
    #combine result to Output[N][K], where N= 0-23, K= number of folds
    rms_all = rms_s + rms_w + rms_k 
    rms_all = np.matrix(rms_all)
    
    #avg all group in each k fold.
    rms_all_folds = []
    for k in range(num_folds):
        rms_all_folds.append(np.mean(rms_all[:,k]))
        
    #Add final result of all k to the list for boxplot
    rms_allParams_Ncategory_Kfolds.append(rms_all)
    rms_all_params.append(rms_all_folds)
    
    rms_all_folds.index(max(rms_all_folds))
###********* full results for analysis
save_object(rms_allParams_Ncategory_Kfolds,'case1_full_rmse.pkl') 

## =============================================================================
## plot graph    
## =============================================================================
print('Ploting result...')
# bigger font size
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

plt.rc('font', **font)


# plot figure
plt.figure(2)
fig, ax = plt.subplots()
#******** Set label here ********
param_all = [20, 25, 30, 35, 40]
bp = ax.boxplot(rms_all_params, labels = param_all, 
                #positions = alphas_all,
                  notch=0, vert=1, whis=1.5)

#ax.set_xlim(0, max(alphas_all)+1)
#ax.set_xticks(alphas_all)
ax.set_xticklabels(param_all) 

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
ax.set_title('LightGBM RMSE over Number of Leaves')
ax.set_xlabel('Number of Leaves')
ax.set_ylabel('RMSE')

plt.legend()
plt.show()

fig.savefig('case1result.png', bbox_inches='tight')





##**case2
score = []
for i,params in enumerate(params_list):
    score.append(np.mean(rms_all_params[i]))
bestParam = score.index(min(score))
best_leaves = params_list[bestParam]["num_leaves"]

lr = [0.01,0.02,0.03,0.04,0.05]
params1 = {}
params1['learning_rate'] = lr[0]
params1['boosting_type'] = 'gbdt'
params1['objective'] = 'regression_l2'
params1['metric'] = 'l2_root'
params1['sub_feature'] = 0.5
params1['num_leaves'] = best_leaves
params1['min_data'] = 50
params1['max_depth'] = 10
params1['is_unbalance'] = True
params1['num_iterations'] = 1000
params1['reg_lambda'] = 0.02

params2 = {}
params2['learning_rate'] = lr[1]
params2['boosting_type'] = 'gbdt'
params2['objective'] = 'regression_l2'
params2['metric'] = 'l2_root'
params2['sub_feature'] = 0.5
params2['num_leaves'] = best_leaves
params2['min_data'] = 50
params2['max_depth'] = 10
params2['is_unbalance'] = True
params2['num_iterations'] = 1000
params2['reg_lambda'] = 0.02

params3 = {}
params3['learning_rate'] = lr[2]
params3['boosting_type'] = 'gbdt'
params3['objective'] = 'regression_l2'
params3['metric'] = 'l2_root'
params3['sub_feature'] = 0.5
params3['num_leaves'] = best_leaves
params3['min_data'] = 50
params3['max_depth'] = 10
params3['is_unbalance'] = True
params3['num_iterations'] = 1000
params3['reg_lambda'] = 0.02

params4 = {}
params4['learning_rate'] = lr[3]
params4['boosting_type'] = 'gbdt'
params4['objective'] = 'regression_l2'
params4['metric'] = 'l2_root'
params4['sub_feature'] = 0.5
params4['num_leaves'] = best_leaves
params4['min_data'] = 50
params4['max_depth'] = 10
params4['is_unbalance'] = True
params4['num_iterations'] = 1000
params4['reg_lambda'] = 0.02

params5 = {}
params5['learning_rate'] = lr[4]
params5['boosting_type'] = 'gbdt'
params5['objective'] = 'regression_l2'
params5['metric'] = 'l2_root'
params5['sub_feature'] = 0.5
params5['num_leaves'] = best_leaves
params5['min_data'] = 50
params5['max_depth'] = 10
params5['is_unbalance'] = True
params5['num_iterations'] = 1000
params5['reg_lambda'] = 0.02


params_list = [params1,params2,params3,params4,params5]

## =============================================================================
## Train and test model to compare performance btw each configurations     
## =============================================================================
rms_allParams_Ncategory_Kfolds = []
rms_all_params = []

for idx,params in enumerate(params_list):
    
    # create 24 x k model for each fold
    for s in range(y_S.shape[1]):
        print(str(idx)+'_Training S'+ str(s) +' model...')
        kfolds_model_S[s],kfolds_x_train_S[s],kfolds_x_test_S[s],kfolds_y_train_S[s],kfolds_y_test_S[s] = create_model_kfolds(num_folds,x_S,y_S[:,s],params)
        
    for w in range(y_W.shape[1]):
        print(str(idx)+'_Training W'+ str(w) +' model...')
        kfolds_model_W[w], kfolds_x_train_W[w], kfolds_x_test_W[w], kfolds_y_train_W[w], kfolds_y_test_W[w] = create_model_kfolds(num_folds,x_W,y_W[:,w],params) 
    
    for k in range(y_K.shape[1]):
        print(str(idx)+'_Training K'+ str(k) +' model...')
        kfolds_model_K[k], kfolds_x_train_K[k], kfolds_x_test_K[k], kfolds_y_train_K[k], kfolds_y_test_K[k] = create_model_kfolds(num_folds,x_K,y_K[:,k],params)

    # Predict each model in S category
    S_predicted = []
    
    for i in range(y_S.shape[1]):
        print(str(idx)+'_Testing S'+ str(i) +' model...')
        model = kfolds_model_S[i]
        x_test = kfolds_x_test_S[i]
        folds_predicted = []
        #predict each fold under each S model (there are 5 models for S)
        for k in range(num_folds):
            y_predicted = model[k].predict(x_test[k])
            folds_predicted.append(y_predicted)
        S_predicted.append(folds_predicted)
        
        
    # Predict each model in W category
    W_predicted = []
    for i in range(y_W.shape[1]):
        print(str(idx)+'_Testing W'+ str(i) +' model...')
        model = kfolds_model_W[i]
        x_test = kfolds_x_test_W[i]
        folds_predicted = []
        #predict each fold under each W model 
        for k in range(num_folds):
            y_predicted = model[k].predict(x_test[k])
            folds_predicted.append(y_predicted)
        W_predicted.append(folds_predicted)
      
    # Predict each model in K category
    K_predicted = []
    for i in range(y_K.shape[1]):
        print(str(idx)+'_Testing K'+ str(i) +' model...')
        model = kfolds_model_K[i]
        x_test = kfolds_x_test_K[i]
        folds_predicted = []
        #predict each fold under each W model 
        for k in range(num_folds):
            y_predicted = model[k].predict(x_test[k])
            folds_predicted.append(y_predicted)
        K_predicted.append(folds_predicted)
        
    # Calculate RMSE    
    rms_s = []    
    rms_w = []
    rms_k = []
    
    for i in range(y_S.shape[1]):
        print(str(idx)+'_Calculate RMSE for S'+ str(i) +' model...')
        folds_rms = []
        for k in range(num_folds):
            rms = sqrt(mean_squared_error(kfolds_y_test_S[i][k],S_predicted[i][k])) 
            folds_rms.append(rms)
        #assign all k-fold result to each model
        rms_s.append(folds_rms)        
        
    for i in range(y_W.shape[1]):
        print(str(idx)+'_Calculate RMSE for W'+ str(i) +' model...')
        folds_rms = []
        for k in range(num_folds):
            rms = sqrt(mean_squared_error(kfolds_y_test_W[i][k],W_predicted[i][k])) 
            folds_rms.append(rms)  
        #assign all k-fold result to each model
        rms_w.append(folds_rms)      
        
    for i in range(y_K.shape[1]):
        print(str(idx)+'_Calculate RMSE for K'+ str(i) +' model...')
        folds_rms = []
        for k in range(num_folds):
            rms = sqrt(mean_squared_error(kfolds_y_test_K[i][k],K_predicted[i][k])) 
            folds_rms.append(rms)    
        #assign all k-fold result to each model
        rms_k.append(folds_rms)        
     
    print('Combining result...')   
    #combine result to Output[N][K], where N= 0-23, K= number of folds
    rms_all = rms_s + rms_w + rms_k 
    rms_all = np.matrix(rms_all)
    
    #avg all group in each k fold.
    rms_all_folds = []
    for k in range(num_folds):
        rms_all_folds.append(np.mean(rms_all[:,k]))
        
    #Add final result of all k to the list for boxplot
    rms_allParams_Ncategory_Kfolds.append(rms_all)
    rms_all_params.append(rms_all_folds)
    
    rms_all_folds.index(max(rms_all_folds))
###********* full results for analysis
save_object(rms_allParams_Ncategory_Kfolds,'case2_full_rmse.pkl') 


    
## =============================================================================
## plot graph    
## =============================================================================
print('Ploting result...')
# bigger font size
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

plt.rc('font', **font)


# plot figure
plt.figure(2)
fig, ax = plt.subplots()
#******** Set label here ********
param_all = [0.01, 0.02, 0.03, 0.04, 0.05]
bp = ax.boxplot(rms_all_params, labels = param_all, 
                #positions = alphas_all,
                  notch=0, vert=1, whis=1.5)

#ax.set_xlim(0, max(alphas_all)+1)
#ax.set_xticks(alphas_all)
ax.set_xticklabels(param_all) 

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
ax.set_title('LightGBM RMSE over Learning rate')
ax.set_xlabel('Learning rate')
ax.set_ylabel('RMSE')

plt.legend()
plt.show()
fig.savefig('case2result.png', bbox_inches='tight')
#
runtime = time.time() - starttime
print('Total runtime: '+ str(runtime))
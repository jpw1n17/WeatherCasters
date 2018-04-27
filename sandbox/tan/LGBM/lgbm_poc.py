import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#====== Prepare data
#dataset = pd.read_csv('../../../code/gencsv/gencsv_ep100_vec100_Kpre/gencsv_ep100_vec100_Kpre.txt', sep = '\t', header = None)
datalen = dataset.shape[1]
outputlen = 24
inputlen = datalen-outputlen

X = dataset[dataset.columns[0:inputlen]].values

result = []
for idx, i in enumerate(range(10,25)):
    print("round: " + str(idx))
    # 's1'[1],'s2','s3','s4','s5','w1','w2','w3','w4'
    #,'k1'[10],'k2','k3','k4','k5','k6','k7','k8','k9','k10','k11','k12','k13','k14','k15']
    #select output to predict. Ex. 10 = k1 , 24 = k15
    selected_output = i #10 
    y = dataset[dataset.columns[inputlen-1+selected_output]].values
    
    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    ## Feature Scaling
    #from sklearn.preprocessing import StandardScaler
    #sc = StandardScaler()
    #x_train = sc.fit_transform(x_train)
    #x_test = sc.transform(x_test)
    
    #======Setup parameters
    import lightgbm as lgb
    d_train = lgb.Dataset(x_train, y_train.flatten())
    #d_train = lgb.Dataset(x_train, y_train.flatten(),categorical_feature=list(range(10)))
    params = {}
    params['learning_rate'] = 0.003
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'regression_l2'
    #params['objective'] = 'binary'
    params['metric'] = 'l2_root'
    #params['metric'] = 'binary_logloss'
    params['sub_feature'] = 0.5
    params['num_leaves'] = 30
    params['min_data'] = 50
    params['max_depth'] = 10
    params['is_unbalance'] = True
    
    
    #======Training model
    clf = lgb.train(params, d_train, 100)
    
    
    #======Prediction
    y_pred=clf.predict(x_test)
    
    
    #======Evaluation
    #RMSE
    import math
    from sklearn.metrics import mean_squared_error
    rmse=math.sqrt(mean_squared_error(y_pred,y_test))
    print("RMSE_"+ str(i) + ":" +str(rmse))
    result.append(rmse)

#RMSE gencsv_ep100_vec10_Kpre :0.12361350112844519
#RMSE gencsv_ep100_vec100_Kpre:0.12316789696891321

#======Compare result
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.title('Actual vs Predicted');plt.xlabel('Actual');plt.ylabel('Predicted')
plt.ylim(0, 1)

#======Compare result distribution
n_bins=100
fig, axs = plt.subplots(1, 2, sharey=True, sharex=False, tight_layout=True)
# We can set the number of bins with the `bins` kwarg
axs[0].hist(y_test, bins=n_bins)
axs[1].hist(y_pred, bins=n_bins)

#plt.subplot(121)
#plt.hist(y_test, bins=n_bins)
#plt.title('Actual')
#plt.ylabel('Frequency')
#plt.suptitle('Histogram Actual vs Predicted', fontsize=16)
#
#plt.subplot(122)
#plt.hist(y_pred, bins=n_bins)
#plt.xlabel('Value')
#plt.title('Predicted')
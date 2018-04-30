# S-Vector

import numpy
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import math
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

#====== Prepare data
dataset = pd.read_csv('../../../code/gencsv/gencsv_ep100_vec100_Spre/gencsv_ep100_vec100_Spre.txt', sep = '\t', header = None)
datalen = dataset.shape[1]
outputlen = 24
inputlen = datalen-outputlen

X = dataset[dataset.columns[0:inputlen]].values

result = []
model = []
predicted = []
actual = []

for idx, i in enumerate(range(1,25)):
    print("Round: " + str(idx))
    # s[1:5]: 's1','s2','s3','s4','s5'
    # w[6:9]: 'w1','w2','w3','w4'
    #,k[10:24]: 'k1','k2','k3','k4','k5','k6','k7','k8','k9','k10','k11','k12','k13','k14','k15']
    #select output to predict. Ex. 10 = k1 , 24 = k15
    selected_output = i #10 
    y = dataset[dataset.columns[inputlen-1+selected_output]].values
    
    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
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
    clf = lgb.train(params, d_train, 100)
    
    
    #======Prediction
    y_pred=clf.predict(x_test)
    
    
    #======Evaluation
    #RMSE
    rmse=math.sqrt(mean_squared_error(y_pred,y_test))
    print("RMSE_"+ str(i) + ":" +str(rmse))
    
    predicted.append(y_pred)
    actual.append(y_test)
    result.append(rmse)

# sample usage
save_object(model, 'all_model_trained_by_S-data.pkl')
save_object(actual, 'actual_by_S-data.pkl')
save_object(predicted, 'predicted_by_S-data.pkl')

print(result)
avg_result = numpy.mean(result)
print("AVG of S-RMSE: "+ str(avg_result))

N = len(result)
x = range(1,N+1)

ax = plt.subplot(111)
ax.set_title('RMSE from model trained by S-pre data')
ax.set_xlabel('output index')
ax.set_xticks(x)
ax.set_ylabel('RMSE')
ax.bar(x, result)
ax.axhline(y=avg_result, color='r', linestyle='-')

#RMSE gencsv_ep100_vec10_Kpre :0.12361350112844519
#RMSE gencsv_ep100_vec100_Kpre:0.12316789696891321

##======Compare result
#import matplotlib.pyplot as plt
#plt.scatter(y_test, y_pred)
#plt.title('Actual vs Predicted (K15)');plt.xlabel('Actual');plt.ylabel('Predicted')
#plt.ylim(0, 1)
#
##======Compare result distribution
#n_bins=20
#fig, axs = plt.subplots(1, 2, sharey=True, sharex=False, tight_layout=True)
## We can set the number of bins with the `bins` kwarg
#axs[0].hist(y_test, bins=n_bins)
#axs[0].set_ylabel('Frequency')
#axs[0].set_title('K15-Actual')
#axs[1].hist(y_pred, bins=n_bins)
#axs[1].set_xlabel('Probability')
#axs[1].set_title('K15-Predicted')

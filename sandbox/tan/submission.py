from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import math
import pickle
from sklearn.metrics import mean_squared_error

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def load_object(path):
    with open(path, 'rb') as input:
        obj = pickle.load(input)
    return obj

# =============================================================================
# read data from csv
# =============================================================================
#train_data = pd.read_csv('gencsv_ep100_vec100_Spre.csv'
#                        ,header=None,sep=',',error_bad_lines=False,encoding='utf-8')

test_data_s = pd.read_csv('../../../code/gencsv_testvectors/gencsv_ep100_vec100_Spre_test_vec.txt'
                        ,header=None,sep='\t',error_bad_lines=False,encoding='utf-8')
test_data_w = pd.read_csv('../../../code/gencsv_testvectors/gencsv_ep100_vec100_Wpre_test_vec.txt'
                        ,header=None,sep='\t',error_bad_lines=False,encoding='utf-8')
test_data_k = pd.read_csv('../../../code/gencsv_testvectors/gencsv_ep100_vec100_Kpre_test_vec.txt'
                        ,header=None,sep='\t',error_bad_lines=False,encoding='utf-8')
#load model
model_s = load_object('all_model_trained_by_S-data.pkl')
model_w = load_object('all_model_trained_by_W-data.pkl')
model_k = load_object('all_model_trained_by_K-data.pkl')

#get id for output
test_orig = pd.read_csv('../../../data/test.csv'  ,sep=',',error_bad_lines=False,encoding='utf-8')
test_row_id = test_orig["id"].astype(int)

#create output as dataframe
output = pd.DataFrame()
output[0] = test_row_id

for i in range(24):
    if(i>=0 and i <=4):
        model = model_s
        test_data = test_data_s
    elif(i>=5 and i <=8):
        model = model_w
        test_data = test_data_w
    else :
        model = model_s
        test_data = test_data_k
        
    result = model[i].predict(test_data)
    output[i+1] = result

hdr ="id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15"

output_format = '%d'

for i in range(24):
    output_format = output_format + ',%.8f'

np.savetxt('lgbm_gencsv_nl_180_lr_007_Pre_S_predict.csv',output,fmt=output_format,delimiter=',',comments='',header=hdr)









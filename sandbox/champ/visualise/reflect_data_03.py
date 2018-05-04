# =============================================================================
# Description
# =============================================================================
# - Add Emoticon eg. :P
# - Include negative modal verb
# - Include ! ?
# - Lemmatize to Verb form

# =============================================================================
# Steps to run the program in the Terminal to reproduce a Doc2Vec model
# =============================================================================
# 1. Open terminal 
# 2. Set environment variable before run Python intepreter
#   > set PYTHONHASHSEED=0
# =============================================================================


# =============================================================================
# Import Packages
# =============================================================================
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import re
import emot
import gensim
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import os

os.chdir('D:\Champ\AnacondaProjects\WeatherCasters\Local\sandbox\champ\\visualise')

# =============================================================================
# Show start time of this program
# =============================================================================
print('start time : '+str(dt.datetime.now()) )

# =============================================================================
# read data from csv
# =============================================================================
def read_train_Spre_data():
    num_data = pd.read_csv('../data/gencsv_ep100_vec100_Spre.csv'
                        ,header=None,sep=',',error_bad_lines=False,encoding='utf-8')      
    return num_data

## =============================================================================
## put data into variables
## =============================================================================
def split_input_output(num_data,n_output):
    
    n_datacol = len(num_data.columns)

    n_input = n_datacol-n_output

    y = num_data.loc[:,n_input:n_input+n_output-1]
    x = num_data.loc[:,0:n_input-1]
    return x,y

def split_train_test(x,y,tr_nrows):
    
    # get the train set
    ytr = y[0:tr_nrows]
    xtr = x[0:tr_nrows]

    # get the test set
    yts = y[tr_nrows:all_nrows]
    xts = x[tr_nrows:all_nrows] 
    
    return xtr,ytr,xts,yts

# =============================================================================
# create model
# =============================================================================
def create_model(x_train, y_train,gamma):
    ridge = linear_model.Ridge(alpha=gamma, fit_intercept=False)
    ridge.fit(x_train, y_train)
    ridge.coef_
    return ridge

# =============================================================================
# Main Program
# =============================================================================

#n_output = 24
#
#num_data = read_train_Spre_data()
#x,y = split_input_output(num_data,n_output)
#
## =============================================================================
## train model
## =============================================================================
#
## get number of rows
#all_nrows = len(x)
#
## get number of training rows
#tr_percent = 0.9
#tr_nrows = round(all_nrows*tr_percent) - 1
#    
#
#xtr,ytr,xts,yts = split_train_test(x,y,tr_nrows)
#
#gamma = 0.2
#
## create and train the Sparse Regression model
#srg = create_model(xtr,ytr,gamma)
#                      
#yhts = srg.predict(xts)
#yhts = pd.DataFrame(yhts)

### =============================================================================
### Plot graph
### =============================================================================
## bigger font size
#font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 14}
#
#plt.rc('font', **font)
#
#
## set number of points in the graph 
##n_point = 1000 
#n_point = all_nrows - tr_nrows 
#n_pt = n_point - 1
#
#t = np.arange(0, 1, 0.1)
#
#plt.figure(1)
#plt.scatter(yts.loc[tr_nrows:tr_nrows+n_pt,123], yhts.loc[0:n_pt,23])
#plt.plot(t, t, color='green')
#plt.show()

# =============================================================================
# Visualize data-points in colour
# =============================================================================
#
colors = np.random.rand(n_point)

plt.figure(2)
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(yts.loc[tr_nrows:tr_nrows+n_pt,123], yhts.loc[0:n_pt,23]
            , c=colors, alpha=0.5)
plt.plot(t, t, color='green')

# Hide these grid behind plot objects
ax.set_axisbelow(True)
ax.set_title('Random Colour')
ax.set_xlabel('Actual value')
ax.set_ylabel('Predicted value')

plt.show()

# =============================================================================
# Visualize data-points in colour by output value
# =============================================================================

###   y.loc[:,123] 
###   This means allo rows of the column 123 (124th column)
###     in the DataFrame y.


# =============================================================================
# color by output value
colors = []
for i in range(n_point):
    ### actual output value
    val = yts.loc[tr_nrows+i,123]
    
    ### predicted output value
    #val = yhts.loc[i,23]
    
    #cl_val =  val
    
    thrh_val = [0.1,0.3,0.5,0.7,0.9]
    if val >= 0 and val < float(thrh_val[0]):
        cl_val = "red"
    elif val >= float(thrh_val[0]) and val < float(thrh_val[1]):
        cl_val = "orange"
    elif val >= float(thrh_val[1]) and val < float(thrh_val[2]):
        cl_val = "yellow"
    elif val >= float(thrh_val[2]) and val < float(thrh_val[3]):
        cl_val = "green"
    elif val >= float(thrh_val[3]) and val < float(thrh_val[4]):
        cl_val = "blue"
    elif val >= float(thrh_val[4]) and val <= 1:
        cl_val = "purple"
    else:
        cl_val = "black"
    
    colors.append(cl_val)
colors = np.asarray(colors)
# =============================================================================



plt.figure(3)
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(yts.loc[tr_nrows:tr_nrows+n_pt,123], yhts.loc[0:n_pt,23]
            , c=colors, alpha=0.5)
plt.plot(t, t, color='green')

# Hide these grid behind plot objects
ax.set_axisbelow(True)
ax.set_title('Colour by Output Value')
ax.set_xlabel('Actual value')
ax.set_ylabel('Predicted value')

plt.show()

# =============================================================================
# 
# =============================================================================

vecdata = xts
#
## Center the data
#docvec_data = vecdata - vecdata.mean()
#
#similarities = 1 - cosine_similarity(docvec_data)
#
## Multidimensional Scaling for Doc2Vec data
#seed = np.random.RandomState(seed=3)
#
#mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
#                   dissimilarity="precomputed", n_jobs=1)
#
#pos = mds.fit(similarities).embedding_
#
#nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
#                    dissimilarity="precomputed", random_state=seed, n_jobs=1,
#                    n_init=1)
#npos = nmds.fit_transform(similarities, init=pos)

# =============================================================================
# 
# =============================================================================







# =============================================================================
# 
# =============================================================================




# =============================================================================
# 
# =============================================================================





# =============================================================================
# Show finish time of this program
# =============================================================================
print('finish time : '+str(dt.datetime.now()) )


# =============================================================================
#     
# =============================================================================



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

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# =============================================================================
# Redirect the directory
# =============================================================================

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

n_output = 24

num_data = read_train_Spre_data()
x,y = split_input_output(num_data,n_output)

# =============================================================================
# train model
# =============================================================================

# get number of rows
all_nrows = len(x)

# get number of training rows
tr_percent = 0.9
tr_nrows = round(all_nrows*tr_percent) - 1
    

xtr,ytr,xts,yts = split_train_test(x,y,tr_nrows)

gamma = 0.2

# create and train the Sparse Regression model
srg = create_model(xtr,ytr,gamma)
                      
yhts = srg.predict(xts)
yhts = pd.DataFrame(yhts)

## =============================================================================
## Plot graph
## =============================================================================
# bigger font size
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

plt.rc('font', **font)


# set number of points in the graph 
#n_point = 1000 
n_point = all_nrows - tr_nrows 
n_pt = n_point - 1

t = np.arange(0, 1, 0.1)

plt.figure(1)
plt.scatter(yts.loc[tr_nrows:tr_nrows+n_pt,123], yhts.loc[0:n_pt,23])
plt.plot(t, t, color='green')
plt.show()

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

# add a grid which has grey color
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

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
# Color code
# =============================================================================

#Red = "#E74C3C"
#Orange = "#EB984E"
#Yellow = "#F4D03F"
#Green = "#2ECC71"
#Blue = "#5DADE2"
#Purple = "#A569BD"
#Grey = "#AEB6BF"
#Black = "#17202A"

col_opt_val = ["#E74C3C","#EB984E","#F4D03F","#2ECC71",
               "#5DADE2","#A569BD","#AEB6BF","#17202A"]

opt_rng = ["0 - 0.1",">0.1 - 0.3",">0.3 - 0.5",">0.5 - 0.7"
               ,">0.7 - 0.9",">0.9 - 1",">1", "0<"]

# =============================================================================
# Setting the color of each point
# =============================================================================
# color by output value
indx_lst = []
colors = []   # list of color
for i in range(n_point):
    ### actual output value
    val = yts.loc[tr_nrows+i,123]
    
    ### predicted output value
    #val = yhts.loc[i,23]
    
    #cl_val =  val
    
    thrh_val = [0.1,0.3,0.5,0.7,0.9]
    if val >= 0 and val < float(thrh_val[0]):
        cl_val = "red"
        grp_indx = 0
        
    elif val >= float(thrh_val[0]) and val < float(thrh_val[1]):
        cl_val = "orange"
        grp_indx = 1
        
    elif val >= float(thrh_val[1]) and val < float(thrh_val[2]):
        cl_val = "yellow"
        grp_indx = 2
        
    elif val >= float(thrh_val[2]) and val < float(thrh_val[3]):
        cl_val = "green"
        grp_indx = 3
        
    elif val >= float(thrh_val[3]) and val < float(thrh_val[4]):
        cl_val = "blue"
        grp_indx = 4
        
    elif val >= float(thrh_val[4]) and val <= 1:
        cl_val = "purple"
        grp_indx = 5
        
    elif val > 1:
        cl_val = "grey"
        grp_indx = 6
        
    else:
        cl_val = "black"
        output_range = "0<"
        grp_indx = 7
        
            
    colors.append(cl_val)
    indx_lst.append(grp_indx)
colors = np.asarray(colors)
# =============================================================================
#  Plot the graph 
# =============================================================================

plt.figure(3)
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(yts.loc[tr_nrows:tr_nrows+n_pt,123], yhts.loc[0:n_pt,23]
            , c=colors, alpha=0.5)
plt.plot(t, t, color='green')


# add a grid which has grey color
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

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

# Center the data
docvec_data = vecdata - vecdata.mean()

similarities = 1 - cosine_similarity(docvec_data)

# Multidimensional Scaling for Doc2Vec data
seed = np.random.RandomState(seed=3)

#mds = manifold.MDS(n_components=2, max_iter=100, eps=1e-9, random_state=seed,
#                   dissimilarity="precomputed", n_jobs=1)

#pos = mds.fit(similarities).embedding_

# Rotate the data
clf = PCA(n_components=2)
pca_docvecs = clf.fit_transform(docvec_data)

#pca_pos = clf.fit_transform(pos)

# transfrom to DataFrame
pca_dv_df = pd.DataFrame(pca_docvecs)



plt.figure(4)
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter( pca_dv_df.loc[:,0] ,pca_dv_df.loc[:,1]
            , c=colors, alpha=0.5)

# add a grid which has grey color
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

ax.set_axisbelow(True)

plt.show()

# =============================================================================
# grouping data by the colour
# =============================================================================

# Create data frame that has the result 
df = pd.DataFrame(dict(x=pca_dv_df.loc[:,0], y=pca_dv_df.loc[:,1], label=indx_lst)) 
# Group by cluster
groups = df.groupby('label')

# =============================================================================
# Visualise in 2D scatter plot
# =============================================================================

plt.figure(5)
fig, ax = plt.subplots(figsize=(10, 6))
for indx, group in groups:
    ax.plot( group.x, group.y
            , marker='o', linestyle='',ms=4 
            , color=col_opt_val[indx]
            , label=opt_rng[indx]
            )
    
# add legend    
plt.legend(scatterpoints=1, loc='best', shadow=False)

# add a grid which has grey color
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

ax.set_axisbelow(True)
ax.set_title('2D Visualisation Coloured by Actual Output Value')
plt.show()



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



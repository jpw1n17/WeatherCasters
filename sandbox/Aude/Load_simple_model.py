# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 14:05:34 2018

@author: audeb
"""

import gensim
import csv
from nltk import RegexpTokenizer
from nltk.corpus import stopwords

model = gensim.models.doc2vec.Doc2Vec.load('simple.model')
#dir(model) => check functions, variables from the model

#using the infer vector function
infer_vec = model.infer_vector(['weather', 'today', 'hot'])
print('infered vector')
print(infer_vec)

docvecs =[]

print('vector in model')
# Append a vector of each tweet
for i in range(0,len(model.docvecs)) :
    twt = model.docvecs[i]
    print (twt)
    docvecs.append(twt)
    
#saving in csv file
    
np.savetxt('Load_simple_model.txt',infer_vec,fmt='%.8f',delimiter='\t', comments='')
    



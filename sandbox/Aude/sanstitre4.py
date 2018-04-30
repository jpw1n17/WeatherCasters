# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:21:04 2018

@author: audeb
"""
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import re
import string
import emot
import gensim
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

test_data = pd.read_csv('../../data/test.csv'
                         ,names = ["id", "tweets", "state", "location"]
                        ,header=0,sep=',',error_bad_lines=False,encoding='utf-8')

for j in range(len(test_data.id)):
    if str(test_data.id[j]).isnumeric() == False:
        print(j)

lst=[]
for twt in test_data.id:
    lst.append(twt)


print( 'number of records : ' + str(len(test_data)) )

np.savetxt('checking_test_data.txt',lst,fmt='%.0f',delimiter='\t', comments='')




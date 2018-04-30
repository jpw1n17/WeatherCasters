### K-Preprocessing Description
# - Include % amd minus for tempurature forcast figures
# - Tag and Keep all verbs, adj, noun, adv

# To reproduce a Doc2Vec model
# Open terminal 
# Need to set environment variable before run Python intepreter
#set PYTHONHASHSEED=0


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


# Show start time of this program
print('start time : '+str(dt.datetime.now()) )

test_data = pd.read_csv('../../data/test.csv'
                         ,names = ["id", "tweets", "state", "location"]
                        ,header=0,sep=',',error_bad_lines=False,encoding='utf-8')

# Need to change the id number to become text to use as tag of documents 
# for example
# 'id'+str(train_data.id[0])
# this simple code will produce a list of ['id1', 'id2', 'id5',...,'id100,...]
tag_id = []
for tid in test_data.id:
    tag_id.append('id'+str(tid)) 

#TODO later: Include emoji about weater in tokenizer ()
tokenized_text = []
#Include % amd minus for tempurature forcast figures
tokenizer = RegexpTokenizer("\w+|%|-")

#Start pre-processing
for tweet in test_data.tweets:
    #Tokenize
    tokens = tokenizer.tokenize(tweet)   
               
    #Pos tagging
    append_pos = []
    tagged_tokens = nltk.pos_tag(tokens)
    for posTag in tagged_tokens: 
        # Tagging is case sensitive, so lower needs to be after
        lower_word = posTag[0].lower()
        
        #Keep all verbs, adj, noun, adv
        if (posTag[1].startswith("V") 
            or posTag[1].startswith("J")
            or posTag[1].startswith("N")
            or posTag[1].startswith("R")) :
            append_pos.append(lower_word)  
            
    #Append each tokenized tweet in the list
    tokenized_text.append(append_pos)
    
      
#=============================================================================

model = gensim.models.doc2vec.Doc2Vec.load('ep100_vec100_Kpre_model.model')

#create a variable in which we append the list of twt inferred from tokenized_text
infer_vectors =[]

print('infered vector')
for twt in tokenized_text:
    infer_twt = model.infer_vector(twt)
    infer_vectors.append(infer_twt)

print( 'number of records : ' + str(len(infer_vectors)) )

np.savetxt('gencsv_ep100_vec100_Kpre_test_vec.txt',infer_vectors,fmt='%.8f',delimiter='\t', comments='')

# =============================================================================
# Show finish time of this program
print('finish time : '+str(dt.datetime.now()) )
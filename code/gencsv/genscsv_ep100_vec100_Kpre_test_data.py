# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:13:54 2018

@author: audeb
"""

# Show start time of this program
print('start time : '+str(dt.datetime.now()) )


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


 # Load a CSV	
test_data = pd.read_csv('../../data/test.csv'
                         ,names = ["id", "tweets", "state", "location"]
                         ,header=0,sep=',',error_bad_lines=False,encoding='utf-8')

# Need to change the id number to become text to use as tag of documents 
# for example
# 'id'+str(test_data.id[0])
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
    
# This iterator returns tagged documents which are tweets with doc label 
class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.TaggedDocument(doc,[self.labels_list[idx]])

# Iterator of tagged documents
iter_tag_doc = LabeledLineSentence(tokenized_text, tag_id)


# Create a Doc2Vec model
model = gensim.models.Doc2Vec(size=100, min_count=0
                              , alpha=0.025, min_alpha=0.025
                              , seed=0, workers=1)

# Build a set of vocabulary
model.build_vocab(iter_tag_doc)
print( 'number of vocabulary : ' + str(len(model.wv.vocab)) )

# Train the doc2vec model
for epoch in range(100):    # number of epoch
 #print( 'iteration '+str(epoch+1) )
 model.train(iter_tag_doc, total_examples=len(tokenized_text), epochs=1 )
 # Change learning rate for next epoch
 model.alpha -= 0.002
 model.min_alpha = model.alpha
print( 'model trained' )


#saving the created model
#model.save('doc2vec.model')
#print( 'model saved' )


docvecs = []

# Append a vector of each tweet
for i in range(0,len(model.docvecs)) :
    twt = model.docvecs[i]
    # print (twt)
    docvecs.append(twt)

    
# Drop to get only output columns
out_data = test_data.drop(columns=["id", "tweets", "state", "location"], axis=1)   
out_data = out_data.values.tolist()


# Create a list of inputs and outputs (numerical data)
features = []
num_data = []

for nn in range(0,len(docvecs)):       
    features = list(docvecs[nn])  
    features.extend(out_data[nn])
    num_data.append(features)


# Write csv file
np.savetxt('gencsv_ep100_vec100_Kpre_test_data.txt',num_data,fmt='%.8f',delimiter='\t', comments='')


# Read numerical data into num_data
#num_data = pd.read_csv('gencsv_ep100_vec100_Spre.txt'
#                        ,header=None,sep='\t',error_bad_lines=False,encoding='utf-8')


# Show finish time of this program
print('finish time : '+str(dt.datetime.now()) )
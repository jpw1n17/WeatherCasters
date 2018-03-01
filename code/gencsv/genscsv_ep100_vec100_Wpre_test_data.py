# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:49:38 2018

@author: SUSMITA
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
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import datetime as dt

# Show start time of this program
print('start time : '+str(dt.datetime.now()) )

 # Load a CSV	
train_data = pd.read_csv(r'A:\sem2\data mining\group project\data\train.csv'
                         ,names = ["id", "tweets", "state", "location", "s1", "s2", "s3", "s4", "s5", "w1", "w2", "w3", "w4", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10", "k11", "k12", "k13", "k14", "k15"]
                        ,header=0,sep=',',error_bad_lines=False,encoding='utf-8')


test_data = pd.read_csv(r'A:\sem2\data mining\group project\data\test.csv'
                         , names = ["id", "tweets", "state", "location"]
                         , header=0, sep=',', error_bad_lines = False, encoding='utf-8')




##### TRAIN PART


# Change the id number to become text to use as tag of documents 
tag_id = []
for tid in train_data.id:
    tag_id.append('id'+str(tid)) 

#Define Emoticon list
baseRegx="\w+"
regx= baseRegx
tokenized_text = []
tokenizer = RegexpTokenizer(regx)

#Define adverb of time and noun of time
#https://www.englishclub.com/vocabulary/adverbs-time.htm
time_adv= ["now","then","today","tomorrow","tonight","yesterday","annually","daily","fortnightly","hourly","monthly","nightly","quarterly","weekly","yearly","always","constantly","ever","frequently","generally","infrequently","never","normally","occasionally","often","rarely","regularly","seldom","sometimes","regularly","usually","already","before","early","earlier","eventually","finally","first","formerly","just","last","late","later","lately","next","previously","recently","since","soon","still","yet"]
# http://www.english-for-students.com/Noun-Words-for-Time.html
time_noun = ["afternoon","age","beginning","calendar","century","clock","date","dawn","day","decade","end","era","evening","forenoon","fortnight","future","hour","midday","midnight","minute","month","morning","night","noon","past","present","previous","season","second","sunrise","sunset","today","tomorrow","week","year","yesterday"]   
time_month= ["january","jan","february","feb","march","mar","april","apr","may","june","jun","july","jul","august","aug","september","sep","sept","october","oct","november","nov","december","dec"]
time_day = ["sunday","sun","monday","mon","tuesday","tue","tues","wednesday","wed","thursday","thu","thur","thurs","friday","fri","saturday","sat"]
time_season = ["spring","summer","autumn","fall","winter"]
time_custom = ["forecast","day","month","year","season"]
time_word_list = time_adv + time_noun + time_month + time_day + time_season + time_custom

#Start pre-processing
for tweet in train_data.tweets:
    #Tokenize
    tokens = tokenizer.tokenize(tweet)   
               
    #Pos tagging
    append_pos = []
    tagged_tokens = nltk.pos_tag(tokens)
    for posTag in tagged_tokens: 
        # Tagging is case sensitive, so lowwer needs to be after
        lower_word = posTag[0].lower()
        
        #Keep all verbs, Modal verb, words of time 
        if (posTag[1].startswith("V") 
            or posTag[1] == "MD" 
            or lower_word in time_word_list) :
            append_pos.append(lower_word)  
            
    #Append each tokenized tweet in the list
    tokenized_text.append(append_pos) 

###############################################################################
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
 model.alpha -= 0.002       # decreasing 0.002 /epoch 
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

    #####
# Drop to get only output columns

out_data = train_data.drop(columns=["id", "tweets", "state", "location"], axis=1)   
out_data = out_data.values.tolist()


# Create a list of inputs and outputs (numerical data)
features = []
num_data = []

for nn in range(0,len(docvecs)):       
    features = list(docvecs[nn])  
    features.extend(out_data[nn])
    num_data.append(features)



######################################### TEST PART #########################################

tag_id_test = []
for tid_test in test_data.id:
    tag_id_test.append('id'+str(tid_test)) 

#Define Emoticon list
baseRegx_test="\w+"
regx_test= baseRegx_test
tokenized_text_test = []
tokenizer_test = RegexpTokenizer(regx_test)

#Start pre-processing
for tweet_test in test_data.tweets:
    #Tokenize
    tokens_test = tokenizer_test.tokenize(tweet_test)   
               
    #Pos tagging
    append_pos_test = []
    tagged_tokens_test = nltk.pos_tag(tokens_test)
    for posTag_test in tagged_tokens_test: 
        # Tagging is case sensitive, so lowwer needs to be after
        lower_word_test = posTag_test[0].lower()
        
        #Keep all verbs, Modal verb, words of time 
        if (posTag_test[1].startswith("V") 
            or posTag_test[1] == "MD" 
            or lower_word_test in time_word_list) :
            append_pos_test.append(lower_word_test)  
            
    #Append each tokenized tweet in the list
    tokenized_text_test.append(append_pos_test) 



############################### PART TO CHECK ###############################

# Infered vector from test set tokenised 
token_test = []
    
for j in (tokenized_text_test):   #docvecs : holds all trained vectors
   # tokenized_test = model.docvecs[j]
    docvec_test = model.infer_vector(j)
    #print (docvec_test)
    token_test.append(docvec_test)

################################sample code#############################
#known_words = "love cute cats".split()
#unknown_words = "astronaut kisses moon".split()
#mixed_words = "the albatross is chicken".split()
#for words in (known_words, unknown_words, mixed_words):
    #v1 = model.infer_vector(words)
    #for i in xrange(100):
        #v2 = model.infer_vector(words)
        #assert np.all(v1 == v2), "Failed on %s" % (" ".join(words))
###############################################################################


# Write csv file
np.savetxt('gencsv_ep100_vec100_Wpre_test_data_2.txt',num_data,fmt='%.8f',delimiter='\t', comments='')

print( 'saved txt file : ' + str(len(num_data)) + ' records')

# Read numberical data into num_data
#num_data = pd.read_csv('gencsv_ep100_vec100_Wpre.txt'
#                        ,header=None,sep='\t',error_bad_lines=False,encoding='utf-8')


# Show finish time of this program
print('finish time : '+str(dt.datetime.now()) )

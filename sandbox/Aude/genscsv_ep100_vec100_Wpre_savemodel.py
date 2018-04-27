### Description
# - Add Emoticon eg. :P
# - Include negative modal verb
# - Include ! ?
# - Lemmatize to Verb form

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

 # Load a CSV	
train_data = pd.read_csv('../../data/train.csv'
                         ,names = ["id", "tweets", "state", "location", "s1", "s2", "s3", "s4", "s5", "w1", "w2", "w3", "w4", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10", "k11", "k12", "k13", "k14", "k15"]
                        ,header=0,sep=',',error_bad_lines=False,encoding='utf-8')

#file_path_1 = 'D:/Champ/AnacondaProjects/WeatherCasters/Local/sandbox/champ/data/simple_01.csv'
#train_data = pd.read_csv(file_path_1
#                         ,names = ["id", "tweets", "state", "location", "s1", "s2", "s3", "s4", "s5", "w1", "w2", "w3", "w4", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10", "k11", "k12", "k13", "k14", "k15"]
#                         ,header=0,sep=',',error_bad_lines=False,encoding='utf-8')

# Need to change the id number to become text to use as tag of documents 
# for example
# 'id'+str(train_data.id[0])
# this simple code will produce a list of ['id1', 'id2', 'id5',...,'id100,...]
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
 model.alpha -= 0.002
 model.min_alpha = model.alpha
print( 'model trained' )


#saving the created model
model.save('doc2vec.model')
print( 'model saved' )

# =============================================================================
# docvecs = []
# 
# # Append a vector of each tweet
# for i in range(0,len(model.docvecs)) :
#     twt = model.docvecs[i]
#     # print (twt)
#     docvecs.append(twt)
# 
#     
# # Drop to get only output columns
# #"s1", "s2", "s3", "s4", "s5", "w1", "w2", "w3", "w4", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10", "k11", "k12", "k13", "k14", "k15"
# out_data = train_data.drop(columns=["id", "tweets", "state", "location"], axis=1)   
# out_data = out_data.values.tolist()
# 
# 
# # Create a list of inputs and outputs (numerical data)
# features = []
# num_data = []
# 
# for nn in range(0,len(docvecs)):       
#     features = list(docvecs[nn])  
#     features.extend(out_data[nn])
#     num_data.append(features)
# 
# 
# # Write csv file
# np.savetxt('gencsv_ep100_vec100_Wpre.txt',num_data,fmt='%.8f',delimiter='\t', comments='')
# 
# print( 'saved txt file : ' + str(len(num_data)) + ' records')
# 
# # Read numberical data into num_data
# #num_data = pd.read_csv('gencsv_ep100_vec100_Wpre.txt'
# #                        ,header=None,sep='\t',error_bad_lines=False,encoding='utf-8')
# 
# 
# =============================================================================
# Show finish time of this program
print('finish time : '+str(dt.datetime.now()) )
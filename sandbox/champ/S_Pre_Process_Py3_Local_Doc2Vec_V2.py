# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:58:29 2018

@author: audeb
"""

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
import numpy


 # Load a CSV	
#train_data = pd.read_csv('../../data/train.csv'
#                         ,names = ["id", "tweets", "state", "location", "s1", "s2", "s3", "s4", "s5", "w1", "w2", "w3", "w4", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10", "k11", "k12", "k13", "k14", "k15"]
#                        ,header=0,sep=',',error_bad_lines=False,encoding='utf-8')

#file_path_1 = 'D:/Champ/AnacondaProjects/WeatherCasters/Local/data/train.csv'
file_path_2 = r'C:\Users\audeb\Southampton Uni\Sem2\Data Mining\Group CW\GitHub\WeatherCasters\sandbox\champ\data/simple_01.csv'

#train_data = pd.read_csv(file_path_1
#                         ,names = ["id", "tweets", "state", "location", "s1", "s2", "s3", "s4", "s5", "w1", "w2", "w3", "w4", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10", "k11", "k12", "k13", "k14", "k15"]
#                         ,header=0,sep=',',error_bad_lines=False,encoding='utf-8')

train_data = pd.read_csv(file_path_2
                         ,names = ["id", "tweets"]
                         ,header=0,sep=',',error_bad_lines=False,encoding='utf-8')

#Define Emoticon list
emolist = list(emot.EMOTICONS.keys())
customEmo = ["^^","^-^","^_^","^ ^",":(","=)"]
emolist.extend(customEmo)

#Include emoticon in tokenizer
baseRegx="\w+|\!|\?"
regx= baseRegx + ""
for emo in emolist: 
    regx = regx + "|" + re.escape(emo)
tokenized_text = []
tokenizer = RegexpTokenizer(regx)

#Get english stopwords
eng_stopwords = stopwords.words('english') 
negative_words = ["aren","aren't","couldn","couldn't","didn","didn't","doesn","doesn't","don","don't","hadn","hadn't","hasn","hasn't","haven","haven't","isn","isn't","mightn","mightn't","mustn","mustn't","needn","needn't","no","nor","not","shan","shan't","should've","shouldn","shouldn't","wasn","wasn't","weren","weren't","won","won't","wouldn","wouldn't"]
stop_words_exclude_neg = list(set(eng_stopwords).difference(negative_words))

#Define Lemmatizer
lemmatizer = WordNetLemmatizer()

#Start pre-processing
for tweet in train_data.tweets:
    #Lowercase
    lower_case = tweet.lower()
    
    #Tokenize
    tokens = tokenizer.tokenize(lower_case)
    
    #Re-initial token list in each round
    filtered_tokens=[] 
    
    #Remove stop word but include the negative helping verb
    for word in tokens:
        if not word in stop_words_exclude_neg:
            #Lemmatize 
            lemmatized = lemmatizer.lemmatize(word, pos="v")
            filtered_tokens.append(lemmatized)
        
    #Append each tokenized tweet in the list
    tokenized_text.append(filtered_tokens)
    
# This iterator returns tagged documents which are tweets with doc label 
class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.TaggedDocument(doc,[self.labels_list[idx]])

# Iterator of tagged documents
iter_tag_doc = LabeledLineSentence(tokenized_text, train_data.id)


# Create a Doc2Vec model
model = gensim.models.Doc2Vec(size=2, min_count=0
                              , alpha=0.025, min_alpha=0.025
                              , seed=0, workers=1)

# Build a set of vocabulary
model.build_vocab(iter_tag_doc)
print( 'number of vocabulary : ' + str(len(model.wv.vocab)) )

# Train the doc2vec model
for epoch in range(10):    # number of epoch
 #print( 'iteration '+str(epoch+1) )
 model.train(iter_tag_doc, total_examples=len(tokenized_text), epochs=1 )
 # Change learning rate for next epoch
 model.alpha -= 0.002
 model.min_alpha = model.alpha
print( 'model trained' )


##saving the created model
##model.save('Pre_S_2_doc2vec.model')
##print( 'model saved' )
#
#
## Test the model
##to get most similar document with similarity scores using document- name
#sims = model.docvecs.most_similar(0, topn=4)
#print('top similar document: ')
#print (sims)
#
##to get vector of document that are not present in corpus 
#docvec = model.infer_vector('I love sunshine.')
#print('infered vector: ')
#print (docvec)
#
## Get vector of tokenized text
#docvec = model.infer_vector(['I', 'love', 'sunshine'])
#print('infered vector: ')
#print (docvec)
#
#
docvecs = []
#
## Display a vector of each tweet
for i in range(0,len(model.docvecs)) :
    docvec = model.docvecs[i]
    docvecs.append(docvec)
    print (docvec)
    #print(docvecs)
#
## Two Dimensions visualisation
#vec_x = []
#vec_y = []
#for i in range(0,len(docvecs)) :
#    vec_x.append(docvecs[i][0])
#    vec_y.append(docvecs[i][1])   
#
## Append the inferred vector
##vec_x.append(docvec[0])
##vec_y.append(docvec[1])   
#
##matplotlib.rcParams['axes.unicode_minus'] = False
#fig, ax = plt.subplots()
#ax.plot(vec_x, vec_y , 'o')
##ax.set_title('title')
#plt.show()
#
#
#
#

# Saving the data to a csv file
numpy.savetxt('np3.csv', docvecs, delimiter=',', header='a,#2')

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
for tweet in test_data.tweets:
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
    
#=============================================================================

model = gensim.models.doc2vec.Doc2Vec.load('ep100_vec100_Spre_model.model')

#create a variable in which we append the list of twt inferred from tokenized_text
infer_vectors =[]

print('infered vector')
for twt in tokenized_text:
    infer_twt = model.infer_vector(twt)
    infer_vectors.append(infer_twt)

print( 'number of records : ' + str(len(infer_vectors)) )

np.savetxt('gencsv_ep100_vec100_Spre_test_vec.txt',infer_vectors,fmt='%.8f',delimiter='\t', comments='')

# =============================================================================
# Show finish time of this program
print('finish time : '+str(dt.datetime.now()) )
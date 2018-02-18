### Description
# - Include % amd minus for tempurature forcast figures
# - Tag and Keep all verbs, adj, noun, adv

import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import re
import string
import emot

 # Load a CSV	
train_data = pd.read_csv('../../data/train.csv'
                         ,names = ["id", "tweets", "state", "location", "s1", "s2", "s3", "s4", "s5", "w1", "w2", "w3", "w4", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10", "k11", "k12", "k13", "k14", "k15"]
                         ,header=0,sep=',',error_bad_lines=False,encoding='utf-8')

#TODO later: Include emoji about weater in tokenizer ()
tokenized_text = []
#Include % amd minus for tempurature forcast figures
tokenizer = RegexpTokenizer("\w+|%|-")

#Start pre-processing
for tweet in train_data.tweets:
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
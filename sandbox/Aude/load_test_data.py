### W preprocessing Description
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
 test_data = pd.read_csv('../../data/test.csv'
                          ,names = ["id", "tweets", "state", "location"]
                         ,header=0,sep=',',error_bad_lines=False,encoding='utf-8')
 
 
 tag_id = []
 for tid in test_data.id:
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
 for tweet in test_data.tweets:
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
 
 #=============================================================================

model = gensim.models.doc2vec.Doc2Vec.load('ep100_vec100_Wpre_model.model')

#create a variable in which we append the list of twt inferred from tokenized_text
infer_vectors =[]

print('infered vector')
for twt in tokenized_text:
    infer_twt = model.infer_vector(twt)
    infer_vectors.append(infer_twt)


np.savetxt('Load_test_build_Wpre.txt',infer_vectors,fmt='%.8f',delimiter='\t', comments='')

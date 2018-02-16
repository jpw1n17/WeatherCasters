#Some used imp
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
import string   # string.punctuation '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~’ 
import emot
#from emoticon import EmojiWordReader

 # Load a CSV	
train_data = pd.read_csv('train.csv'
                         ,names = ["id", "tweets", "state", "location", "s1", "s2", "s3", "s4", "s5", "w1", "w2", "w3", "w4", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10", "k11", "k12", "k13", "k14", "k15"]
                         ,header=0,sep=',',error_bad_lines=False,encoding='utf-8')

#Assign variables

#Don'y care Emoticon

baseRegx="\w+|\!|\?"
regx= baseRegx
tokenized_text = []
tokenizer = RegexpTokenizer(regx)

#Get english stopwords
eng_stopwords = stopwords.words('english') 

#Define Lemmatizer
lemmatizer = WordNetLemmatizer()

#Start pre-processing
for tweet in train_data.tweets:
    #Lowercase
    lower_case = tweet.lower()
    
    #####Note
    #- tuple[1] start with "V"
    #- modal verb "MD"
    #- custom "adv of time", "point of time",
    #words = ["last","yet","tomorrow","yesterday", "next", "is", "am", "were","go","gone","going", "can", "might", "can't", "couldn't" ]
    #pos = nltk.pos_tag(words)
    ######
    
    #Tokenize
    tokens = tokenizer.tokenize(lower_case)
    
    #Re-initial token list in each round
    filtered_tokens=[] 
    
    #Remove stop word
    # ?should we remove stopword? see row 20, 21 as "can't" can tell something about sentiment
    for word in tokens:
        if not word in eng_stopwords:
            #Lemmatize 
            lemmatized = lemmatizer.lemmatize(word, pos="v")
            filtered_tokens.append(lemmatized)
        
    #Append each tokenized tweet in the list
    tokenized_text.append(filtered_tokens)
    #tokenized_text
    
#Tagging

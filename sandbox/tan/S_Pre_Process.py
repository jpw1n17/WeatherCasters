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
train_data = pd.read_csv('../../data/train.csv'
                         ,names = ["id", "tweets", "state", "location", "s1", "s2", "s3", "s4", "s5", "w1", "w2", "w3", "w4", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10", "k11", "k12", "k13", "k14", "k15"]
                         ,header=0,sep=',',error_bad_lines=False,encoding='utf-8')

#Assign variables

#Define Emoticon list
emolist = emot.EMOTICONS.keys()
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
    #tokenized_text
    
#Tagging


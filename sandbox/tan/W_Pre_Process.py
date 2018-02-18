### Description
# - Tag and Keep Verb, Modal verb, 
# - Include custome words related to time (adv of time, noun of time)
# - No lemmatize to prevent tense changing
# - Lemmatize to Verb form
#**Remark
# - after "to" if the word no meaning and lowercase will be verb
 
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
#from emoticon import EmojiWordReader

 # Load a CSV	
train_data = pd.read_csv('../../data/train.csv'
                         ,names = ["id", "tweets", "state", "location", "s1", "s2", "s3", "s4", "s5", "w1", "w2", "w3", "w4", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10", "k11", "k12", "k13", "k14", "k15"]
                         ,header=0,sep=',',error_bad_lines=False,encoding='utf-8')

#Don't care Emoticon
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
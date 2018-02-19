# -*- coding: utf-8 -*-
"""
 reproduce doc2vec model
"""
# Open terminal 
# Need to set environment variable before run Python intepreter
#set PYTHONHASHSEED=0


# To set PYTHONHASHSEED after open intepreter using os.environ is not work.
import os
os.environ["PYTHONHASHSEED"] = "0"
#print(os.environ["PYTHONHASHSEED"])

# Import all the dependencies
import gensim
import csv
from nltk import RegexpTokenizer
from nltk.corpus import stopwords

#import numpy as np
#import matplotlib
import matplotlib.pyplot as plt

# Download stopwords
#import nltk
#nltk.download('stopwords')
stopword_set = set(stopwords.words('english'))

# Create variables
train_data = []
docvecs = []

# Set file path
file_path = './data/simple_01.csv'

# Read File
with open(file_path,encoding="utf8") as csv_file:
    reader = csv.reader(csv_file)
    train_data = [row for row in reader]

# Separate header out from the dataset
header = train_data[0]
train_data = train_data[1:len(train_data)] 

# Extract doc label from the dataset
docLabels = [row[0] for row in train_data]

# Extract tweet data from the dataset
docData = [row[1] for row in train_data]

# Create a tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# This function does all cleaning of data using tokenizer and stopwords
def nlp_clean(data):
   new_data = []
   for d in data:
      new_str = d.lower()
      dlist = tokenizer.tokenize(new_str)
      dlist = list(set(dlist).difference(stopword_set))
      new_data.append(dlist)
   return new_data

# Clean data
data = nlp_clean(docData)

# This iterator returns tagged documents which are tweets with doc label 
class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.TaggedDocument(doc,[self.labels_list[idx]])

# Iterator of tagged documents
iter_tag_doc = LabeledLineSentence(data, docLabels)

# Create a Doc2Vec model
model = gensim.models.Doc2Vec(size=2, min_count=0, alpha=0.025, min_alpha=0.025, seed=0, workers=1)

# Build a set of vocabulary
model.build_vocab(iter_tag_doc)
print( 'number of vocabulary : ' + str(len(model.wv.vocab)) )

# Train the doc2vec model
for epoch in range(10):    # number of epoch
 #print( 'iteration '+str(epoch+1) )
 model.train(iter_tag_doc, total_examples=len(data), epochs=1 )
 # Change learning rate for next epoch
 model.alpha -= 0.002
 model.min_alpha = model.alpha
print( 'model trained' )

docvecs = []

# Display a vector of each tweet
for i in range(0,len(model.docvecs)) :
    docvec = model.docvecs[i]
    docvecs.append(docvec)
    #print (docvec)

#for i in range(0,2) :
#    print (i)

for i in range(0,len(docvecs)) :
    print (docvecs[i])
    
# Check reproducibility of doc2vec model
#[ 0.02201098 -0.16212247]
#[0.02891408 0.18884255]


# Two Dimensions visualisation
vec_x = []
vec_y = []
for i in range(0,len(docvecs)) :
    vec_x.append(docvecs[i][0])
    vec_y.append(docvecs[i][1])   

#matplotlib.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots()
ax.plot(vec_x, vec_y , 'o')
#ax.set_title('title')
plt.show()






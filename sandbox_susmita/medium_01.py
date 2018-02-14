# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:54:52 2018

@author: SUSMITA
"""

#Import all the dependencies
import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join

train_data = []

with open('A:/sem2/data mining/group project/data/train.csv',encoding="utf8") as file:
    # Read File
    reader = csv.reader(file)
    train_data = [row for row in reader]


tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))
#This function does all cleaning of data using two objects above
def nlp_clean(data):
   new_data = []
   for d in data:
      new_str = d.lower()
      dlist = tokenizer.tokenize(new_str)
      dlist = list(set(dlist).difference(stopword_set))
      new_data.append(dlist)
   return new_data


class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc,    
[self.labels_list[idx]])

header = train_data[0]

train_data = train_data[1:len(train_data)] 

docLabels = [row[0] for row in train_data]

data = [row[1] for row in train_data]

data = nlp_clean(data)

#iterator returned over all documents
it = LabeledLineSentence(data, docLabels)


model = gensim.models.Doc2Vec(size=300, min_count=0, alpha=0.025, min_alpha=0.025)
model.build_vocab(it)

#training of model
for epoch in range(1):
 print( 'iteration '+str(epoch+1) )
 model.train(it, total_examples=len(data), epochs=1 )
 model.alpha -= 0.002
 model.min_alpha = model.alpha
 model.train(it, total_examples=len(data), epochs=1)
#saving the created model
model.save('doc2vec.model')
print( 'model saved' )



#loading the model
d2v_model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model')

d2v_model = model

#start testing
#printing the vector of document at index 1 in docLabels
docvec = d2v_model.docvecs[1]
print (docvec)

#printing the vector of the file using its name
docvec = d2v_model.docvecs[data[0]] #if string tag used in training
print (docvec)

#to get most similar document with similarity scores using document-index
similar_doc = d2v_model.docvecs.most_similar(14) 
print (similar_doc)

#to get most similar document with similarity scores using document- name
sims = d2v_model.docvecs.most_similar(data[0])
print (sims)

#to get vector of document that are not present in corpus 
docvec = d2v_model.docvecs.infer_vector(data[0])
print (docvec)





# -*- coding: utf-8 -*-
"""
 reproduce doc2vec model
"""

# Import all the dependencies
import gensim
import csv
from nltk import RegexpTokenizer
from nltk.corpus import stopwords

# Download stopwords
#import nltk
#nltk.download('stopwords')
stopword_set = set(stopwords.words('english'))

# Create variables
train_data = []

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
model = gensim.models.Doc2Vec(vector_size=2, min_count=0, alpha=0.025, min_alpha=0.025)

# Build a set of vocabulary
model.build_vocab(iter_tag_doc)
print( 'number of vocabulary : ' + str(len(model.wv.vocab)) )

# Train the doc2vec model
for epoch in range(10):    # number of epoch
 print( 'iteration '+str(epoch+1) )
 model.train(iter_tag_doc, total_examples=len(data), epochs=1 )
 # Change learning rate for next epoch
 model.alpha -= 0.002
 model.min_alpha = model.alpha
print( 'model trained' )

#saving the created model
model.save('simple.model')
print( 'model saved' )

# Display a vector of each tweet
for i in range(0,len(model.docvecs)) :
    docvec = model.docvecs[i]
    print (docvec)

#for i in range(0,2) :
#    print (i)

# Check reproducibility of doc2vec model
#[-0.20974638 -0.12714435]
#[-0.19671175 -0.20221564]




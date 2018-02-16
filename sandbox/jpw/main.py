from os import makedirs
from os.path import exists
import pandas as pd
import copy

from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
import sklearn
import matplotlib.pyplot as plt

#global variables
g_data_path = '../../data/'
g_output_path = './gen_data/'

# asimple class for holding all state and passing it around
# for now we will just have a dataframe and dyamically add other properties.
# later we may choose to formalise the class 
class AppState(object):
    def __init__(self, data_frame):
        self.df = data_frame

def add_cleaned_tweets(state):
    print('cleaning tweets')
    tokenizer = RegexpTokenizer(r'\w+')
    stopword_set = set(stopwords.words('english'))
    state.cleaned_tweets = []
    for d in state.df.tweet:
        new_str = d.lower()
        ordered_list = tokenizer.tokenize(new_str)
        clipped_list = list(set(ordered_list).difference(stopword_set)) # side effect of changing word order
        clipped_ordered_list = [word for word in ordered_list if word in clipped_list]
        state.cleaned_tweets.append(clipped_ordered_list)

def add_link_doc_to_id(state):
    ids = state.df.id
    raw_docs = state.df.tweet
    docs = state.cleaned_tweets
    
    state.lookup = {}
    state.lookup_raw = {}
    state.tagged_docs = []
    for doc, raw, id in zip(docs, raw_docs, ids):
        state.lookup[id]=doc
        state.lookup_raw[id]=raw
        state.tagged_docs.append(gensim.models.doc2vec.TaggedDocument(doc, [id]))
   
def load_or_create_vector_space(state):
    vec_space_model_path = g_output_path + 'doc2vec.model'
    if exists(vec_space_model_path):
        print('Loading doc2vec model')
        state.d2vm = gensim.models.Doc2Vec.load(vec_space_model_path)
    else:
        # create from scratch
        print('Creating doc2vec model')
        state.d2vm = gensim.models.Doc2Vec(min_count=0, alpha=0.025, min_alpha=0.025)
        state.d2vm.build_vocab(tagged_docs)
        print('    created vocab')
        state.d2vm.train(data.bag['tagged_docs'], total_examples=len(data.bag['tagged_docs']), epochs=100)
        print('    trained model')
        state.d2vm.save(vec_space_model_path)

def d2v_inspection(state):
    print('vectors space trained, most similar too "strom" are...')
    for item in state.d2vm.wv.most_similar(positive=['storm'], topn=4):
        print(item)

    print('\nvectors space trained, most similar too "storm" while suppresing "wind" are...')
    for item in state.d2vm.wv.most_similar(positive=['storm'], negative=['wind'], topn=4):
        print(item)

    print('\nvectors space trained, most similar too "storm" while suppresing "rain" are...')
    for item in state.d2vm.wv.most_similar(positive=['storm'], negative=['rain'], topn=4):
        print(item)

    test_str = 'looks like another sunny day tomorrow'
    print('\n\ntweets similar to "' +test_str+ '" are ...')
    test_vec = state.d2vm.infer_vector(test_str.split())
    for item in state.d2vm.docvecs.most_similar([test_vec], topn=3):
        tweet_id = item[0]
        likelihood = item[1]
        tweet_text = state.lookup_raw[tweet_id]
        print(str(likelihood) + '\t' + tweet_text)

def main():
    makedirs(g_output_path, exist_ok=True)

    print('Loading source data')
    state = AppState(pd.read_csv(g_data_path + 'train.csv'))
    add_cleaned_tweets(state)
    add_link_doc_to_id(state)
    load_or_create_vector_space(state)
    d2v_inspection(state)

# start of main script 
main()

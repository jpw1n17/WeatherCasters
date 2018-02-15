from os import makedirs
from os.path import exists
import pandas as pd
import copy

from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import gensim

#global variables
g_data_path = '../../data/'
g_output_path = './gen_data/'

class CustomDataFrame(pd.DataFrame):
    bag = {}
    def __init__(self, *args, **kw):
        super(CustomDataFrame, self).__init__(*args, **kw)
 
    @property
    def _constructor(self):
        return CustomDataFrame

def add_cleaned_tweets(data):
    print('cleaning tweets')
    tokenizer = RegexpTokenizer(r'\w+')
    stopword_set = set(stopwords.words('english'))
    data.bag['cleaned_tweets'] = []
    for d in data.tweet:
        new_str = d.lower()
        ordered_list = tokenizer.tokenize(new_str)
        clipped_list = list(set(ordered_list).difference(stopword_set)) # side effect of changing word order
        clipped_ordered_list = [word for word in ordered_list if word in clipped_list]
        data.bag['cleaned_tweets'].append(clipped_ordered_list)

def add_link_doc_to_id(data):
    ids = data.id
    docs = data.bag['cleaned_tweets']
    raw_docs = data.tweet
    data.bag['lookup'] = {}
    data.bag['lookup_raw'] = {}
    data.bag['tagged_docs'] = []
    for doc, raw, id in zip(docs, raw_docs, ids):
        data.bag['lookup'][id]=doc
        data.bag['lookup_raw'][id]=raw
        data.bag['tagged_docs'].append(gensim.models.doc2vec.TaggedDocument(doc, [id]))
   
def load_or_create_vector_space(data):
    d2v_model = {}
    vec_space_model_path = g_output_path + 'doc2vec.model'
    if exists(vec_space_model_path):
        print('Loading doc2vec model')
        return gensim.models.Doc2Vec.load(vec_space_model_path)
    else:
        # create from scratch
        print('Creating doc2vec model')
        d2v_model = gensim.models.Doc2Vec(min_count=0, alpha=0.025, min_alpha=0.025)
        d2v_model.build_vocab(tagged_docs)
        print('    created vocab')
        d2v_model.train(data.bag['tagged_docs'], total_examples=len(data.bag['tagged_docs']), epochs=100)
        print('    trained model')
        d2v_model.save(vec_space_model_path)
        return d2v_model

def d2v_inspection(model, data):
    print('vectors space trained, most similar too "strom" are...')
    for item in model.wv.most_similar(positive=['storm'], topn=4):
        print(item)

    print('\nvectors space trained, most similar too "storm" while suppresing "wind" are...')
    for item in model.wv.most_similar(positive=['storm'], negative=['wind'], topn=4):
        print(item)

    print('\nvectors space trained, most similar too "storm" while suppresing "rain" are...')
    for item in model.wv.most_similar(positive=['storm'], negative=['rain'], topn=4):
        print(item)

    test_str = 'looks like another sunny day tomorrow'
    print('\n\ntweets similar to "' +test_str+ '" are ...')
    test_vec = model.infer_vector(test_str.split())
    for item in model.docvecs.most_similar([test_vec], topn=3):
        tweet_id = item[0]
        likelihood = item[1]
        tweet_text = data.bag['lookup_raw'][tweet_id]
        print(str(likelihood) + '\t' + tweet_text)

def main():
    makedirs(g_output_path, exist_ok=True)

    # print(gensim.__version__)

    print('Loading source data')
    data = CustomDataFrame(pd.read_csv(g_data_path + 'train.csv'))
    add_cleaned_tweets(data)
    add_link_doc_to_id(data)
    d2vm = load_or_create_vector_space(data)
    d2v_inspection(d2vm, data)

# start of main script 
main()

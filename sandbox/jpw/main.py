from os import makedirs
from os.path import exists
import pandas as pd

from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import gensim

#global variables
g_data_path = '../../data/'
g_output_path = './gen_data/'

def nlp_clean(data):
    tokenizer = RegexpTokenizer(r'\w+')
    stopword_set = set(stopwords.words('english'))
    new_data = []
    for d in data:
        new_str = d.lower()
        ordered_list = tokenizer.tokenize(new_str)
        clipped_list = list(set(ordered_list).difference(stopword_set)) # side effect of changing word order
        clipped_ordered_list = [word for word in ordered_list if word in clipped_list]
        new_data.append(clipped_ordered_list)
    return new_data

def link_doc_to_id(docs, ids):
    linked_docs = []
    if len(docs) != len(ids):
        raise Exception('must have same number of docs as ids')
    for doc, id in zip(docs, ids):
        linked_docs.append(gensim.models.doc2vec.TaggedDocument(doc, [id]))
    return linked_docs

def load_or_create_vector_space(docs, ids):
    if len(docs) != len(ids):
        raise Exception('docs and ids must be of same length')
    
    d2v_model = {}
    vec_space_model_path = g_output_path + 'doc2vec.model'
    if exists(vec_space_model_path):
        print('Loading doc2vec model')
        return gensim.models.Doc2Vec.load(vec_space_model_path)
    else:
        # create from scratch
        print('Creating doc2vec model')
        tagged_docs = link_doc_to_id(docs, ids)
        d2v_model = gensim.models.Doc2Vec(min_count=0, alpha=0.025, min_alpha=0.025)
        d2v_model.build_vocab(tagged_docs)
        print('    created vocab')
        d2v_model.train(tagged_docs, total_examples=len(tagged_docs), epochs=100)
        print('    trained model')
        d2v_model.save(vec_space_model_path)
        return d2v_model

def main():
    makedirs(g_output_path, exist_ok=True)

    print('Loading source data')
    #source_data = load_csv(g_data_path + 'train.csv')
    source_data = pd.read_csv(g_data_path + 'train.csv')

    print('cleaning tweets')
    clean_tweets = nlp_clean(source_data['tweet'])

    d2vm = load_or_create_vector_space(clean_tweets, source_data['id'])

    print('vectors space trained, most similar too "strom" are...')
    for item in d2vm.wv.most_similar(positive=['storm'], topn=4):
        print(item)

    print('\nvectors space trained, most similar too "storm" while suppresing "wind" are...')
    for item in d2vm.wv.most_similar(positive=['storm'], negative=['wind'], topn=4):
        print(item)

    print('\nvectors space trained, most similar too "storm" while suppresing "rain" are...')
    for item in d2vm.wv.most_similar(positive=['storm'], negative=['rain'], topn=4):
        print(item)

    test_str = 'looks like another sunny day tomorrow'
    print('\n\ntweets similar to "' +test_str+ '" are ...')
    test_vec = d2vm.infer_vector(test_str.split())
    for tweet in d2vm.docvecs.most_similar([test_vec], topn=3):
        print(tweet)

# start of main script 
main()

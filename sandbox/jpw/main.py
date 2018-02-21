from os import makedirs
from os.path import exists
import pandas as pd
import copy
import numpy as np

import gensim
import sklearn
import matplotlib.pyplot as plt

import preprocess.clean
import regression.linear

#global variables
g_data_path = '../../data/'
g_output_path = './gen_data/'

def index_to_tag(index):
    return 'index_' + str(index)

def tag_to_index(tag):
    return int(tag[6:]) # strip index_

def get_cleaned_tweets(df):
    print('cleaning tweets')
    return pd.Series(preprocess.clean.gen_tokens(df.tweet.values), index=df.index)

def get_doc_tags(docs_series):
    tagged_docs = []
    for index, doc in docs_series.iteritems():
        tagged_docs.append(gensim.models.doc2vec.TaggedDocument(doc, [index_to_tag(index)]))
    return tagged_docs
   
def load_or_create_vector_space(tagged_docs, vector_size):
    d2vm = {}
    vec_space_model_path = g_output_path + 'doc2vec.model'
    if exists(vec_space_model_path):
        print('Loading doc2vec model')
        d2vm = gensim.models.Doc2Vec.load(vec_space_model_path)
    else:
        # create from scratch
        print('Creating doc2vec model')
        d2vm = gensim.models.Doc2Vec(vector_size=vector_size, min_count=0, alpha=0.025, min_alpha=0.025)
        d2vm.build_vocab(tagged_docs)
        print('    created vocab')
        d2vm.train(tagged_docs, total_examples=len(tagged_docs), epochs=100)
        print('    trained model')
        # d2vm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        d2vm.save(vec_space_model_path)
    return d2vm

def similar_doc(d2vm, text):
    vec = d2vm.infer_vector(preprocess.clean.clean_str(text))
    return d2vm.docvecs.most_similar([vec], topn=1)[0]

def d2v_inspection(d2vm):
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
    print('\n\ntweet similar to "' +test_str+ '" are ...')
    index, likelyhood = similar_doc(d2vm, test_str)
    print(str(likelyhood) + ' ' + index)
    
def split_data_frame_train_test(df):
    # for now we want a reproduecable random selection
    np.random.seed(0)
    selection_mask = np.random.rand(len(df)) < 0.8 # 80/20 split
    return df[selection_mask], df[~selection_mask]

# RMSE = sqrt(sum(i= 1 to n, (pi - ai)^2)/n)
# n is 24 times the total number of tweets
# pi is the predicted confidence rating for a given label
# ai is the actual confidence rating for a given label
# 
# moving divide by n into sum to prevent overflow
def evaluate_RMSE_hand_cranked(state, pred):
    sum_error = 0.0
    total_elements = 24 * len(state.ts)
    # confidence values in columns 4 to 27
    source_data_output_col_offset = 4
    for r in range(len(state.ts)):
        for c in range(0, 23):
            pi = pred.iloc[:, c + 1] # first col id
            ai = state.ts.iloc(r, c + source_data_output_col_offset)
            sum_error = sum_error + (((pi - ai)**2) / total_elements)
    return sum_error**(0.5)

def evaluate_RMSE(tr_df, pred_df):
    if len(tr_df) != len(pred_df):
        raise Exception('inconsistent lengths')
    piv = pred_df.iloc[:, 0:24].values
    aiv = tr_df.iloc[:, 4:28].values
    pis = np.concatenate(piv)
    ais = np.concatenate(aiv)
    return sklearn.metrics.mean_squared_error(pis, ais)

def infer_vector_space(raw_doc, d2vm):
    dvs = []
    for doc in raw_doc:
        dvs.append(d2vm.infer_vector(preprocess.clean.clean_str(doc)))
    return dvs

def load_training_csv(path):
    # need to force types on the numeric columns for compatibility with sklearn libraries
    column_types = {'id':np.str, 'tweet':np.str, 'state':np.str, 'location':np.str, 
        's1':np.float, 's2':np.float, 's3':np.float, 's4':np.float, 's5':np.float,
        'w1':np.float, 'w2':np.float, 'w3':np.float, 'w4':np.float,
        'k1':np.float, 'k2':np.float, 'k3':np.float, 'k4':np.float, 'k5':np.float,
        'k6':np.float, 'k7':np.float, 'k8':np.float, 'k9':np.float, 'k10':np.float, 
        'k11':np.float, 'k12':np.float, 'k13':np.float, 'k14':np.float, 'k15':np.float
    }
    return pd.read_csv(path, dtype=column_types)

def extract_doc_vetors(df, d2vm):
    vecs = []
    for index, dummy_row in df.iterrows():
        vecs.append(d2vm.docvecs[index_to_tag(index)])
    return vecs

def main():
    makedirs(g_output_path, exist_ok=True)

    print('Loading source data')
    df = load_training_csv(g_data_path + 'train.csv')
    tr, ts = split_data_frame_train_test(df)
    tr_doc_tags = get_doc_tags(get_cleaned_tweets(tr))
    vector_size = 100
    d2vm = load_or_create_vector_space(tr_doc_tags, vector_size)
    
    # train the regression model
    tr_doc_vecs = extract_doc_vetors(tr, d2vm)
    lrms = regression.linear.train_model(tr, tr_doc_vecs)

    # test the model
    ts_dvs = infer_vector_space(ts.tweet.values, d2vm)
    pred_df = regression.linear.make_predictions(lrms, ts, ts_dvs)
    print('linear regression error ' + str(evaluate_RMSE(ts, pred_df)))

# start of main script 
main()

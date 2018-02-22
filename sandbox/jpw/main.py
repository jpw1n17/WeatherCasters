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
g_confidence_values_offset = 4
g_n_confidence_colums = 24

condifence_descriptions = {
    "s1":"Sentiment - I can't tell",
    "s2":"Sentiment - Negative",
    "s3":"Sentiment - Neutral / author is just sharing information",
    "s4":"Sentiment - Positive",
    "s5":"Sentiment - Tweet not related to weather condition",
    "w1":"Weather tense - current (same day) weather",
    "w2":"Weather tense - future (forecast)",
    "w3":"Weather tense - I can't tell",
    "w4":"Weather tense - past weather",
    "k1":"Weather state - clouds",
    "k2":"Weather state - cold",
    "k3":"Weather state - dry",
    "k4":"Weather state - hot",
    "k5":"Weather state - humid",
    "k6":"Weather state - hurricane",
    "k7":"Weather state - I can't tell",
    "k8":"Weather state - ice",
    "k9":"Weather state - other",
    "k10":"Weather state - rain",
    "k11":"Weather state - snow",
    "k12":"Weather state - storms",
    "k13":"Weather state - sun",
    "k14":"Weather state - tornado",
    "k15":"Weather state - wind"
}

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
# n is g_n_confidence_colums times the total number of tweets
# pi is the predicted confidence rating for a given label
# ai is the actual confidence rating for a given label
# 
# moving divide by n into sum to prevent overflow

def flatten_data_frame(df, c1, c2):
    nmpy_array = df.iloc[:, c1:c2].values
    return np.concatenate(nmpy_array)

def flatten_results(ts_df, pred_df):
    ais = flatten_data_frame(ts_df, g_confidence_values_offset, g_confidence_values_offset + g_n_confidence_colums)
    pis = flatten_data_frame(pred_df, 0, g_n_confidence_colums)
    return ais, pis

def evaluate_RMSE(ts_df, pred_df):
    return sklearn.metrics.mean_squared_error(*flatten_results(ts_df, pred_df))**0.5

def gen_graphs(base_title, tr_df, pred_df):
    graph_dir = g_output_path + 'graphs/'
    makedirs(graph_dir , exist_ok=True)
    headings = tr_df.columns.values
    for c in range(0,g_n_confidence_colums):
        col_name = headings[c + g_confidence_values_offset]
        actual = tr_df.iloc[:, c + g_confidence_values_offset].values
        pred = pred_df.iloc[:,c]
        plt.clf()
        plt.scatter(actual, pred)
        plt.title(base_title + '\n' + col_name + ':' + condifence_descriptions[col_name])
        plt.xlabel('Actual')
        plt.xlim(0.0, 1.0)
        plt.ylabel('Prediction')
        plt.ylim(-1.0, 1.0)
        # plot ideal fit guide line
        plt.plot((0,1), 'r--')
        plt.savefig(graph_dir + col_name)
    plt.close()

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
    gen_graphs('Linear Regression', ts, pred_df)
    print('linear regression RMSE ' + str(evaluate_RMSE(ts, pred_df)))
   
# start of main script 
main()

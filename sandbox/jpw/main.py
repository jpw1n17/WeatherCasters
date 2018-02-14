import csv
from os import listdir
from os import makedirs
from os.path import isfile, isdir, exists, join

from nltk import RegexpTokenizer
from nltk.corpus import stopwords

import gensim

def load_csv(file_path):
    data_obj = {}
    headers = []
    with open(file_path, 'r', encoding='utf8') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        headers = next(reader)
        for col in headers:
            data_obj[col] = []
        for row in reader:
            for col, val in zip(headers, row):
                data_obj[col].append(val)
    data_obj['headers'] = headers
    return data_obj

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

def create_dir_if_not_exist(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)

# start of main script
data_path = '../../data/'
output_path = './gen_data/'
tweet_id_header_col_index = 0
tweet_header_col_index = 1
create_dir_if_not_exist(output_path)


source_data = load_csv(data_path + 'train.csv')
headers = source_data['headers']
tweet_header_label = headers[tweet_header_col_index]
tweet_id_header_label = headers[tweet_header_col_index]

tweet_ids = source_data[tweet_id_header_label]
raw_tweets = source_data[tweet_header_label]
clean_tweets = nlp_clean(raw_tweets)
print('data loaded and cleaned')
linked_tweets = link_doc_to_id(clean_tweets, tweet_ids)
print('data linked')
model = gensim.models.Doc2Vec(min_count=0, alpha=0.025, min_alpha=0.025)
model.build_vocab(linked_tweets)
print('vocab built')

# get the initial document vector, and most similar articles
# (before training, the results should be wrong)
docvec1 = list(model.docvecs[0])
docvecsyn1 = model.docvecs.doctag_syn0[0]
docsim1 = model.docvecs.most_similar(tweet_ids[0])

#training of model
model.train(linked_tweets, total_examples=len(linked_tweets), epochs=100)
print('vectors space trained')

# get the trained document vector, and most similar articles
# (after training, the results should be correct)
docvec2 = model.docvecs[0]
docvecsyn2 = model.docvecs.doctag_syn0[0]
docsim2 = model.docvecs.most_similar(tweet_ids[0])

# print results
# document vector
print('Document vector:')

# before training
print('(Before training)')
print('number of dimensions' + str(len(docvec1)))
print(docvec1[:5])
print(docvecsyn1[:5])

print('(After training)')
print(docvec2[:5])
print(docvecsyn2[:5])

# most similar documents
print('\nMost similar:')

# before training, the result is wrong. after training, correct. good.
print('(Before training)')
print(docsim1[:2])

print('(After training)')
print(docsim2[:2])

model.save(output_path + 'doc2vec.model')
print('model saved')

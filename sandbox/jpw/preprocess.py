# simple file for preprossesing of text
from nltk import RegexpTokenizer
from nltk.corpus import stopwords

def process(original_list):
    tokenizer = RegexpTokenizer(r'\w+')
    stopword_set = set(stopwords.words('english'))
    cleaned_list = []
    for d in original_list:
        new_str = d.lower()
        ordered_list = tokenizer.tokenize(new_str)
        clipped_list = list(set(ordered_list).difference(stopword_set)) # side effect of changing word order
        clipped_ordered_list = [word for word in ordered_list if word in clipped_list]
        cleaned_list.append(clipped_ordered_list)
    return cleaned_list
# simple file for preprossesing of text
from nltk import RegexpTokenizer
from nltk.corpus import stopwords

_tokenizer = RegexpTokenizer(r'\w+')
_stopword_set = set(stopwords.words('english'))

def gen_tokens(original_list):
    cleaned_list = []
    for d in original_list:
        cleaned_list.append(clean_str(d))
    return cleaned_list

def clean_str(text):
    new_str = text.lower()
    ordered_list = _tokenizer.tokenize(new_str)
    clipped_list = list(set(ordered_list).difference(_stopword_set)) # side effect of changing word order
    clipped_ordered_list = [word for word in ordered_list if word in clipped_list]
    return clipped_ordered_list
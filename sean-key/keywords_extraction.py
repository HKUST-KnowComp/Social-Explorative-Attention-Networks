import re
import string
import operator
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer

punct_re = re.compile('[{}]'.format(re.escape(string.punctuation)))
stop_words = set(stopwords.words('english'))


def generate_candidate(text):
    words_ = list()
    words = word_tokenize(text)
    words = list(map(lambda s: s.lower(), words))
    words_.append(words)
    return words_


def extract_tags(text, min_df=5, max_df=0.8, num_key=5):
    vocabulary = [generate_candidate(text)]
    vocabulary = list(chain(*vocabulary))
    vocabulary = list(np.unique(vocabulary))  # unique vocab
    max_vocab_len = max(map(lambda s: len(s.split(' ')), vocabulary))
    tfidf_model = TfidfVectorizer(vocabulary=vocabulary, lowercase=True,
                                  ngram_range=(1, max_vocab_len), stop_words=None,
                                  min_df=min_df, max_df=max_df)
    X = tfidf_model.fit_transform([text])
    vocabulary_sort = [v[0] for v in sorted(tfidf_model.vocabulary_.items(),
                                            key=operator.itemgetter(1))]
    sorted_array = np.fliplr(np.argsort(X.toarray()))

    # return list of top candidate phrase
    key_phrases = list()
    for sorted_array_doc in sorted_array:
        key_phrase = [vocabulary_sort[e] for e in sorted_array_doc[0:num_key]]
        key_phrases.append(key_phrase)

    return key_phrases

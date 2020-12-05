"""Contains functionalities to preprocess and clean data before using"""
import os

import regex
from nltk import FreqDist
from nltk.tokenize import word_tokenize

from sentiment import utils
from sentiment.services.features import get_word_features


def remove_stop_words(tokens):
    """
    Removes stop words in token list.
    Langauge setting default to `english`.
    :param tokens: list of tokens
    :return: list of tokens
    """
    stop_words = utils.load_stop_word_list('english')
    return [t for t in tokens if not t in stop_words]


def remove_punctuations(text):
    p = regex.compile(r'\p{P}+')
    return p.sub(' ', text)


def get_word_freq_dict(data_path):
    """
    Returns a word frequency dict (Bag of Words) using the text data provided.

    :param data_path: path to data
    :return: nltk FreqDist obj, e.g.
    FreqDist({'good': 13703, 'bad': 8461, 'great': 8297, ...}]
    """
    clean_tokens = []
    for f in os.listdir(data_path):
        if f.endswith('.txt'):
            fh = open(os.path.join(data_path, f), 'r').read()
            no_punctuations = remove_punctuations(fh)
            tokens = word_tokenize(no_punctuations)
            clean_tokens.extend([t.lower() for t in remove_stop_words(tokens)])

    return FreqDist(clean_tokens)


def find_features(text):
    """
    Find whether the top word features exist in the given input text.

    :param text: string
    :return: features as dict, e.g.
    {'good': True, 'silly': False, 'rock': False, ...}
    """
    top_word_features = get_word_features()

    words = word_tokenize(text)
    features = dict()
    for w in top_word_features:
        features[w] = (w in words)

    return features


def split_train_test_dataset(feature_sets, train_test_split=0.8):
    """

    :param feature_sets: a list of tuples
    :param train_test_split: defaulted to 80%:20%
    :return:
    """
    total = len(feature_sets)
    train = feature_sets[:int(total * train_test_split)]
    test = feature_sets[int(total * train_test_split):]
    return train, test

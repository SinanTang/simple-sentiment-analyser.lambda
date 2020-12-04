"""Loads pre-pickled data and feature sets"""
import os
import pickle

from sentiment import utils


def get_labeled_sentiment_data():
    """
    Load pre-pickled sentiment data with labels.
    """
    pickle_file = os.path.join(utils.get_project_root(),
                               'data/pickles/sentiment_data_labeled.pickle')
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)


def get_word_features():
    """
    Returns the top word features in training data.
    Use the existing pickled data.

    :param num_feat: number of features to consider, defaulted to 5k.
    :return: a list of top features (words)
    """
    pickle_file = os.path.join(utils.get_project_root(),
                               'data/pickles/word_features_5k.pickle')
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)


def get_feature_sets():
    """
    Load pre-pickled feature sets data.
    Feature sets structure: a list of tuples: e.g.
    [({'good': True, 'silly': False, ...}, 'pos'),
    ({'good': False, 'silly': True, ...}, 'neg'), (...)]

    :return: Feature sets.
    """
    pickle_file = os.path.join(utils.get_project_root(),
                               'data/pickles/feature_sets.pickle')
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)
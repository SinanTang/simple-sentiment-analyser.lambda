"""Contains functionalities to pickle data and features"""
import os
import pickle
import random

from sentiment import utils
from sentiment.services.preprocess import get_word_freq_dict, find_features
from sentiment.utils import get_project_root


# TODO add data preprocessing
def get_labeled_sentiment_dataset(train_data_dir=utils.get_training_data_path()):
    """
    Label training data with sentiment label, 'pos' or 'neg'.
    :param train_data_dir: default to existing training data.
    :return: [('some_positive_text', 'pos'), ('some_negative_text', 'neg')]
    """
    dataset_with_label = []

    for datafile in os.listdir(train_data_dir):
        if datafile.startswith('positive'):
            for line in open(os.path.join(train_data_dir, datafile)).read().split('\n'):
                dataset_with_label.append((line, 'pos'))
        elif datafile.startswith('negative'):
            for line in open(os.path.join(train_data_dir, datafile)).read().split('\n'):
                dataset_with_label.append((line, 'neg'))

    return dataset_with_label


def pickle_labeled_sentiment_dataset():
    dataset_with_label = get_labeled_sentiment_dataset()

    pickle_as = os.path.join(get_project_root(), 'data/pickles/sentiment_data_labeled_v1.pickle')
    with open(pickle_as, 'wb') as f:
        pickle.dump(dataset_with_label, f)


def pickle_word_features(num_feat=5000):
    default_training_data = utils.get_training_data_path()
    word_dict = get_word_freq_dict(default_training_data)
    features = list(word_dict.keys())[:num_feat]

    pickle_as = os.path.join(get_project_root(), 'data/pickles/word_features_5k.pickle')
    with open(pickle_as, 'wb') as f:
        pickle.dump(features, f)


def pickle_feature_sets():
    dataset_with_label = get_labeled_sentiment_dataset()
    # a list of tuples: e.g. [({'good': True, 'silly': False, ...}, 'pos'),
    # ({'good': False, 'silly': True, ...}, 'neg'), (...)]
    feature_sets = [(find_features(text), sentiment) for (text, sentiment) in dataset_with_label]
    random.shuffle(feature_sets)

    pickle_as = os.path.join(get_project_root(), 'data/pickles/feature_sets.pickle')
    with open(pickle_as, 'wb') as f:
        pickle.dump(feature_sets, f)


if __name__ == "__main__":
    pickle_labeled_sentiment_dataset()
    pickle_word_features()
    pickle_feature_sets()

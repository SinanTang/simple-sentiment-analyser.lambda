"""Loads pre-pickled classifiers"""
import os
import pickle

from sentiment import utils


def load_naive_bayes_clf():
    pickle_clf = os.path.join(utils.get_project_root(),
                              'data/classifiers/naive_bayes_5k.pickle')
    with open(pickle_clf, 'rb') as f:
        return pickle.load(f)


def load_mnb_clf():
    pickle_clf = os.path.join(utils.get_project_root(),
                              'data/classifiers/mnb_classifier_5k.pickle')
    with open(pickle_clf, 'rb') as f:
        return pickle.load(f)


def load_bernoulli_nb_clf():
    pickle_clf = os.path.join(utils.get_project_root(),
                              'data/classifiers/bernoulli_nb_classifier_5k.pickle')
    with open(pickle_clf, 'rb') as f:
        return pickle.load(f)


def load_logistic_regression_clf():
    pickle_clf = os.path.join(utils.get_project_root(),
                              'data/classifiers/logistic_regression_classifier_5k.pickle')
    with open(pickle_clf, 'rb') as f:
        return pickle.load(f)


def load_linear_svc_clf():
    pickle_clf = os.path.join(utils.get_project_root(),
                              'data/classifiers/linear_svc_classifier_5k.pickle')
    with open(pickle_clf, 'rb') as f:
        return pickle.load(f)


def load_all_classifier_pickles():
    return load_naive_bayes_clf(), \
           load_mnb_clf(), \
           load_bernoulli_nb_clf(), \
           load_logistic_regression_clf(), \
           load_linear_svc_clf()

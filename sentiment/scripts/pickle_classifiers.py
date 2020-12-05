"""Contains functionalities to train and pickle classifiers for sentiment classification
using default training data."""
import os
import pickle

from nltk import SklearnClassifier, NaiveBayesClassifier
from nltk import classify
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC

from sentiment import utils
from sentiment.services.preprocess import split_train_test_dataset
from sentiment.services.features import get_feature_sets


def train_naive_bayes_clf(training_set, testing_set):
    """
    accuracy: 74.26
    """
    naive_bayes_classifier = NaiveBayesClassifier.train(training_set)
    print('Naive Bayes model accuracy:',
          (classify.accuracy(naive_bayes_classifier, testing_set)) * 100)

    naive_bayes_classifier.show_most_informative_features(15)

    pickle_as = os.path.join(utils.get_project_root(),
                             'data/classifiers/naive_bayes_5k.pickle')
    with open(pickle_as, 'wb') as f:
        pickle.dump(naive_bayes_classifier, f)


def train_mnb_clf(training_set, testing_set):
    """
    accuracy: 73.28
    """
    mnb_classifier = SklearnClassifier(MultinomialNB())
    mnb_classifier.train(training_set)
    print("Multinomial NB Classifier accuracy:",
          (classify.accuracy(mnb_classifier, testing_set)) * 100)

    pickle_as = os.path.join(utils.get_project_root(),
                             'data/classifiers/mnb_classifier_5k.pickle')
    with open(pickle_as, 'wb') as f:
        pickle.dump(mnb_classifier, f)


def train_bernoulli_nb_clf(training_set, testing_set):
    """
    accuracy: 74.64
    """
    bernoulli_nb_classifier = SklearnClassifier(BernoulliNB())
    bernoulli_nb_classifier.train(training_set)
    print("Bernoulli NB Classifier accuracy:",
          (classify.accuracy(bernoulli_nb_classifier, testing_set)) * 100)

    pickle_as = os.path.join(utils.get_project_root(),
                             'data/classifiers/bernoulli_nb_classifier_5k.pickle')
    with open(pickle_as, 'wb') as f:
        pickle.dump(bernoulli_nb_classifier, f)


def train_logistic_regression_clf(training_set, testing_set):
    """
    accuracy: 74.59
    """
    logistic_regression_classifier = SklearnClassifier(LogisticRegression())
    logistic_regression_classifier.train(training_set)
    print('Logistic Regression Classifier accuracy:',
          (classify.accuracy(logistic_regression_classifier, testing_set)) * 100)

    pickle_as = os.path.join(utils.get_project_root(),
                             'data/classifiers/logistic_regression_classifier_5k.pickle')
    with open(pickle_as, 'wb') as f:
        pickle.dump(logistic_regression_classifier, f)


def train_linear_svc_clf(training_set, testing_set):
    """
    accuracy: 72.01
    """
    linear_svc_classifier = SklearnClassifier(LinearSVC())
    linear_svc_classifier.train(training_set)
    print("LinearSVC Classifier accuracy:",
          (classify.accuracy(linear_svc_classifier, testing_set)) * 100)

    pickle_as = os.path.join(utils.get_project_root(),
                             'data/classifiers/linear_svc_classifier_5k.pickle')
    with open(pickle_as, 'wb') as f:
        pickle.dump(linear_svc_classifier, f)


if __name__ == "__main__":
    feature_sets = get_feature_sets()
    training_set, testing_set = split_train_test_dataset(feature_sets)

    # relatively quick to train
    train_naive_bayes_clf(training_set, testing_set)

    train_mnb_clf(training_set, testing_set)

    train_linear_svc_clf(training_set, testing_set)

    train_bernoulli_nb_clf(training_set, testing_set)

    train_logistic_regression_clf(training_set, testing_set)

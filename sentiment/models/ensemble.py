from statistics import mode

from nltk import ClassifierI


# TODO refactor this class to improve reusability
class EnsembleClassifier(ClassifierI):
    """An ensemble of classifiers is a set of classifiers whose individual decisions
    are combined in some way (typically by weighted or unweighted voting)
    to classify new examples. """

    def __init__(self, classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for clf in self._classifiers:
            v = clf.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        return choice_votes / len(votes)

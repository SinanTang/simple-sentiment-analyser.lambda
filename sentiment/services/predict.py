from sentiment.models.ensemble import EnsembleClassifier
from sentiment.models.load_classifiers import load_all_classifier_pickles
from sentiment.services.preprocess import find_features


def predict_sentiment(text):
    classifiers = load_all_classifier_pickles()
    ensemble_classifier = EnsembleClassifier(classifiers)
    feats = find_features(text)

    return ensemble_classifier.classify(feats), \
           ensemble_classifier.confidence(feats)

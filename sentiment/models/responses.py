def build_response_body(sentiment_prediction, confidence):
    """
    Returns a formatted dict containing sentiment prediction.
    :param sentiment_prediction:
    :param confidence:
    :return:
    """
    return dict(sentiment='{}'.format(sentiment_prediction),
                confidence=confidence)

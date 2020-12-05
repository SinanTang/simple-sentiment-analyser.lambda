import logging

from sentiment.models import requests
from sentiment.models.responses import build_response_body
from sentiment.services.predict import predict_sentiment

LOGGER = logging.getLogger('sentiment.lambda.sentiment_analyser')
LOGGER.setLevel(logging.INFO)


def handler(event, _):
    """
    Entry point for AWS Lambda. In this function we validate the request,
    and build an HTTP response.
    :param event: A dictionary containing parameters
    :param _: Lambdas take two params, event and context, but we don't use context yet,
    so naming it _ as per PEP8
    See https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html
    :return: A json string containing an HTTP statusCode, headers and a body
            OR errors.
    """
    try:
        LOGGER.info("Received request: {}".format(event))
        text = requests.validate_request(event)
        sentiment_prediction, confidence = predict_sentiment(text)

        return dict(statusCode=200,
                    headers={},
                    body=build_response_body(sentiment_prediction,
                                             confidence))

    except Exception as e:
        return dict(statusCode=500,
                    headers={},
                    body={'exception': '{}'.format(e)})

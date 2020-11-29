import json


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
    return dict(statusCode=200,
                headers={},
                body=json.dumps(event))

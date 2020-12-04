from sentiment.models.errors import InvalidEventError


def validate_request(event):
    """
    Validates lambda invocation event and returns the input text string,
    else raises.
    :param event:
    :return: string
    :raises InvalidEventError
    """
    if 'input' in event and len(event['input']) > 1:
        return event['input']
    else:
        raise InvalidEventError('Lambda event does not contain valid input field!')

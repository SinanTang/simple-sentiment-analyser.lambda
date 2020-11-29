# simple-sentiment-analyser.lambda
A simple Python application to detect text sentiment. 

This repo  is for educational purposes. It contains source code to be deployed as a AWS lambda function.

### Development

Ensure the correct version of [Python](https://wiki.python.org/moin/BeginnersGuide/Download) and [`pipenv`](https://github.com/pypa/pipenv) are installed.

Clone the repo:
```shell
git clone git@github.com:SinanTang/simple-sentiment-analyser.lambda.git
cd simple-sentiment-analyser.lambda
```

Create a Pipenv virtual env and install packages:
```shell
pipenv --python 3.8.5
pipenv install --dev
```

### Tests

Run unit tests:
```shell
pipenv run tests
```

### Code Style

[Pylint](https://www.pylint.org/) is used for static code analysis. Run it locally:

```shell
pipenv run lint
```

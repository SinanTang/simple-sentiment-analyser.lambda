import os
from pathlib import Path


def get_project_root():
    return Path(__file__).parent


def get_training_data_path():
    return os.path.join(get_project_root(), 'data/train')


def load_stop_word_list(lang):
    fp = os.path.join(get_project_root(), 'data/stopwords/{}'.format(lang))
    return open(fp, 'r').read().split('\n')

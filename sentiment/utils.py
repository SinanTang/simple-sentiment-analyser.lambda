import os
from pathlib import Path


def get_project_root():
    return Path(__file__).parent


def get_training_data_path():
    return os.path.join(get_project_root(), 'data/train')

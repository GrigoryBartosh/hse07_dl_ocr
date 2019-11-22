import os

__all__ = ['PATH']

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CUR_DIR, '..')

PATH = {
    'DATASETS': {
        # TODO
    },
    'TF_LOGS': os.path.join(ROOT_DIR, 'data', 'logs_tf'),
    'MODELS': {
        'HARDTEXT_DIR': os.path.join(ROOT_DIR, 'data', 'models', 'hardtext'),
    }
}

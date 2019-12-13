import os

__all__ = ['PATH']

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CUR_DIR, '..')

PATH = {
    'DATASETS': {
        'COCO': {
            'TRAIN': {
                'IMAGES_DIR': os.path.join(ROOT_DIR, 'data', 'datasets', 'coco', 'images', 'train2014'),
                'CAPTIONS': os.path.join(ROOT_DIR, 'data', 'datasets', 'coco', 'annotations', 'captions_train2014.json')
            },
            'VAL': {
                'IMAGES_DIR': os.path.join(ROOT_DIR, 'data', 'datasets', 'coco', 'images', 'val2014'),
                'CAPTIONS': os.path.join(ROOT_DIR, 'data', 'datasets', 'coco', 'annotations', 'captions_val2014.json')
            }
        }
    },
    'TF_LOGS': os.path.join(ROOT_DIR, 'data', 'logs_tf'),
    'MODELS': {
        'HARDTEXT_DIR': os.path.join(ROOT_DIR, 'data', 'models', 'hardtext'),
    }
}

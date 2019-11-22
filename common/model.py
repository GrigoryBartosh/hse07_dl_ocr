import os

import torch
import torch.nn as nn

from common.config import PATH
import common.utils as utils

from models.generator import Generator
from models.stacker import Stacker
from models.ocr import OCR

__all__ = ['FullModelWrapper']

ALL_MODEL_DIR = PATH['MODELS']['HARDTEXT_DIR']
MODEL_STATE_EXT = '.pth'


def save_model(model, path, params=None):
    torch.save({
            'params': params,
            'model_state_dict': model.state_dict()
        }, path)


def load_model(model_type, path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    params = checkpoint['params']
    model = model_type(params)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, params


class FullModelWrapper():
    def __init__(self, args, path=None, load=False, device='cpu'):
        self.path = path if path else os.path.join(ALL_MODEL_DIR, 'model')

        self.gen_type = Generator if args['gen'] else None
        self.stacker_type = Stacker if args['stacker'] else None
        self.ocr_type = OCR if args['ocr'] else None

        self.model_names = ['gen', 'stacker', 'ocr']

        for model_name in self.model_names:
            ldict = {'self': self}
            exec(f'model_type = self.{model_name}_type', {}, ldict)
            if ldict['model_type']:
                if load:
                    model, model_params = load_model(
                        ldict['model_type'],
                        os.path.join(self.path, model_name + MODEL_STATE_EXT),
                        device=device
                    )
                else:
                    model_params = args[model_name]
                    model = ldict['model_type'](model_params)
            else:
                model_params = None
                model = None
            ldict['model_params'] = model_params
            ldict['model'] = model
            exec(f'self.{model_name}_params = model_params', {}, ldict)
            exec(f'self.{model_name} = model', {}, ldict)

    def _apply(self, foo):
        for model_name in self.model_names:
            ldict = {'self': self}
            exec(f'model = self.{model_name}', {}, ldict)
            exec(f'model_params = self.{model_name}_params', {}, ldict)
            if ldict['model']:
                foo(ldict['model'], ldict['model_params'], model_name)

    def get_all_models(self):
        models = []
        foo = lambda model, params, name: models.append(model)
        self._apply(foo)
        return models

    def save(self):
        utils.remove_dir(self.path)
        utils.make_dir(self.path)
        foo = lambda model, params, name: save_model(
                model,
                os.path.join(self.path, name + MODEL_STATE_EXT),
                params
            )
        self._apply(foo)

    def to(self, device):
        foo = lambda model, params, name: model.to(device)
        self._apply(foo)

    def train(self):
        foo = lambda model, params, name: model.train()
        self._apply(foo)

    def eval(self):
        foo = lambda model, params, name: model.eval()
        self._apply(foo)
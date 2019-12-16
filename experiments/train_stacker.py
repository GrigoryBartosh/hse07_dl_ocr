import os

import numpy as np

import torch
import torch.optim as optim
import torchvision

from common.config import PATH
import common.utils as utils
from common.data_loading import get_loader
import common.logging as logging
from common.model import save_model, load_model

import common.losses as losses

from models.stacker import Stacker

ALL_MODEL_DIR = PATH['MODELS']['HARDTEXT_DIR']
STACKER_NAME = 'stacker'
MODEL_STATE_EXT = '.pth'
STACKER_PATH = os.path.join(ALL_MODEL_DIR, STACKER_NAME + MODEL_STATE_EXT)

image_size = 300
params_count = 8
params_move_count = 5
args = {
    'load_model': True,
    'model': {
        'encoder': {
            'block': 'ResBasicBlock',
            'layers': [2, 2, 2, 2],
            'activ': 'relu'
        },
        'decoder': {
            'image_size': image_size,
            'block': 'ResBasicBlock',
            'layers': [2, 2, 2, 2],
            'activ': 'lrelu'
        },
        'params_move_count': params_move_count
    },
    'train': { # TODO
        'image_size': image_size,
        'params_count': params_count,
        'params_move_count': params_move_count,
        'save_iter': 50000,
        'val_iter': 50000,
        'val_iter_count': 2048,
        'batch_size': 48,
        'num_workers': 16,
        'lr': 0.0001,
        'w_l2_norm': 0,
        'log_images_count': 5
    }
}


class Trainer():
    def __init__(self, model, args, device, summary_writer):
        self.model = model
        self.args = args
        self.device = device
        self.summary_writer = summary_writer

        self.criterion = losses.StackerLoss()

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args['lr'],
            weight_decay=self.args['w_l2_norm']
        )

        self.all_val_losses = []

    def log_losses(self, phase, losses, n_iter):
        self.summary_writer.add_scalar(
            f'{phase}/reconstruction', losses[0], n_iter)
        self.summary_writer.add_scalar(
            f'{phase}/total', losses[1], n_iter)

    def log_images(self, phase, x, target, params_move, n_iter):
        n = self.args['log_images_count']
        x, target, params_move = x[:n], target[:n], params_move[:n]

        with torch.no_grad():
            out = self.model.move(x, params_move)

        img_grid = torchvision.utils.make_grid(
            torch.cat((x[:, None, :, :], target[:, None, :, :], out[:, None, :, :])),
            normalize=True,
            nrow=x.size(0)
        )
        self.summary_writer.add_image(f'{phase}_recon', img_grid, n_iter)

    def calc_losses(self, x, params_move, target):
        out = self.model.move(x, params_move)

        loss_recon = self.criterion(out, target)

        loss_total = loss_recon

        losses = np.array([
            loss_recon.item(),
            loss_total.item()
        ])

        return losses, loss_total

    def step(self, batch, n_iter):
        _, (_, x, target, params, _, _) = batch
        params_move = params[:, :self.args['params_move_count']]

        x = x.to(self.device)
        params_move = params_move.to(self.device)
        target = target.to(self.device)

        self.optimizer.zero_grad()
        losses, loss_total = self.calc_losses(x, params_move, target)
        loss_total.backward()
        self.optimizer.step()

        self.log_losses('Train', losses, n_iter)

    def eval(self, batch, train_n_iter, val_n_iter):
        _, (_, x, target, params, _, _) = batch
        params_move = params[:, :self.args['params_move_count']]

        x = x.to(self.device)
        params_move = params_move.to(self.device)
        target = target.to(self.device)

        with torch.no_grad():
            losses, _ = self.calc_losses(x, params_move, target)

            self.all_val_losses += [losses]

            if val_n_iter >= self.args['val_iter_count']:
                self.log_losses(
                    'Validation',
                    np.stack(self.all_val_losses, axis=1).mean(axis=1),
                    train_n_iter
                )

                self.all_val_losses = []

                self.log_images('Validation', x, target, params_move, train_n_iter)


if __name__ == '__main__':
    image_size = args['train']['image_size']
    batch_size = args['train']['batch_size']
    num_workers = args['train']['num_workers']
    train_data_loader = get_loader(
        sset='TRAIN',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_data_loader = get_loader(
        sset='VAL',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args['load_model']:
        model, _ = load_model(Stacker, STACKER_PATH)
    else:
        model = Stacker(args['model'])
    model.to(device)

    summary_writer = logging.get_summary_writer()
    utils.write_arguments(summary_writer, args)

    trainer = Trainer(model, args['train'], device, summary_writer)

    model.train()
    iter_last_save = 0
    iter_last_val = 0
    for batch_num, batch in enumerate(train_data_loader, 1):
        batch_size = args['train']['batch_size']
        iter_num = batch_num * batch_size

        trainer.step(batch, iter_num)

        iter_last_save += batch_size
        iter_last_val += batch_size

        if iter_last_save >= args['train']['save_iter']:
            save_model(model, STACKER_PATH, args['model'])
            iter_last_save = 0

        if iter_last_val >= args['train']['val_iter']:
            model.eval()

            for val_batch_num, val_batch in enumerate(val_data_loader, 1):
                val_iter_num = val_batch_num * batch_size

                trainer.eval(val_batch, iter_num, val_iter_num)

                if val_iter_num >= args['train']['val_iter_count']:
                    break

            model.train()
            iter_last_val = 0
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
    'load_model': False, # TODO
    'model': {
        'mover': {
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
            'mlp': {
                'layers': [params_move_count, 16, 32],
                'activ': 'lrelu',
                'dropout_rate': 0
            }
        },
        'dis': {
            'block': 'ResBasicBlock',
            'layers': [2, 2, 2, 2],
            'activ': 'relu'
        },
        'params_move_count': params_move_count
    },
    'train': {
        'image_size': image_size,
        'params_count': params_count,
        'params_move_count': params_move_count,
        'save_iter': 20000,
        'val_iter': 20000,
        'val_iter_count': 4000,
        'batch_size': 40,
        'num_workers': 16,
        'lr': 0.0001,
        'w_recon_auto': 0,
        'w_dis': 0.1,
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

        self.criterion_stacker = losses.StackerLoss()
        self.criterion_dis = losses.DiscriminatorLoss()

        self.optimizer_gen = optim.Adam(
            filter(lambda p: p.requires_grad, model.mover.parameters()),
            lr=self.args['lr'],
            weight_decay=self.args['w_l2_norm']
        )

        self.optimizer_dis = optim.Adam(
            filter(lambda p: p.requires_grad, model.discriminator.parameters()),
            lr=self.args['lr'],
            weight_decay=self.args['w_l2_norm']
        )

        self.all_val_gen_losses = []
        self.all_val_dis_losses = []

    def log_gen_losses(self, phase, losses, n_iter):
        self.summary_writer.add_scalar(
            f'{phase}_gen/auto_reconstruction', losses[0], n_iter)
        self.summary_writer.add_scalar(
            f'{phase}_gen/reconstruction', losses[1], n_iter)
        self.summary_writer.add_scalar(
            f'{phase}_gen/dis', losses[2], n_iter)
        self.summary_writer.add_scalar(
            f'{phase}_gen/total', losses[3], n_iter)

    def log_dis_losses(self, phase, losses, n_iter):
        self.summary_writer.add_scalar(
            f'{phase}_dis/dis', losses[0], n_iter)
        self.summary_writer.add_scalar(
            f'{phase}_dis/total', losses[1], n_iter)

    def log_images(self, phase, x, target, params_move, n_iter):
        n = self.args['log_images_count']
        x, target, params_move = x[:n], target[:n], params_move[:n]

        with torch.no_grad():
            zero_params_move = torch.zeros_like(params_move, device=self.device)
            out = self.model.move(
                torch.cat((x, x), 0),
                torch.cat((zero_params_move, params_move), 0)
            )
            out_auto, out = out[:batch_size], out[batch_size:]

        img_grid = torchvision.utils.make_grid(
            torch.cat((
                x[:, None, :, :],
                target[:, None, :, :],
                out_auto[:, None, :, :],
                out[:, None, :, :],
            )),
            normalize=True,
            nrow=x.size(0)
        )
        self.summary_writer.add_image(f'{phase}_recon', img_grid, n_iter)

    def calc_gen_losses(self, x, params_move, target):
        if self.args['w_recon_auto'] == 0:
            out = self.model.move(x, params_move)

            loss_recon_auto = 0
            loss_recon = self.criterion_stacker(out, target)
        else:
            zero_params_move = torch.zeros_like(params_move, device=self.device)
            batch_size = x.shape[0]
            out = self.model.move(
                torch.cat((x, x), 0),
                torch.cat((zero_params_move, params_move), 0)
            )

            loss_recon_auto = self.criterion_stacker(out[:batch_size], x)
            loss_recon = self.criterion_stacker(out[batch_size:], target)

        out_dis = self.model.dis(out)

        loss_dis = self.criterion_dis(out_dis, type=0)

        loss_total = (loss_recon_auto * self.args['w_recon_auto'] + \
                      loss_recon * (1 - self.args['w_recon_auto'])) * (1 - self.args['w_dis']) + \
                     loss_dis * self.args['w_dis']

        losses = np.array([
            loss_recon_auto.item() if type(loss_recon_auto) != int else loss_recon_auto,
            loss_recon.item(),
            loss_dis.item(),
            loss_total.item()
        ])

        return losses, loss_total

    def calc_dis_losses(self, x, params_move, target):
        out = self.model.move(x, params_move)

        out_dis_0 = self.model.dis(target)
        out_dis_1 = self.model.dis(out)

        loss_dis = 0.5 * (self.criterion_dis(out_dis_0, type=0) + \
                          self.criterion_dis(out_dis_1, type=1))

        loss_total = loss_dis

        losses = np.array([
            loss_dis.item(),
            loss_total.item()
        ])

        return losses, loss_total

    def step(self, batch, n_iter):
        _, (_, x, target, params, _, _) = batch
        params_move = params[:, :self.args['params_move_count']]

        x = x.to(self.device)
        params_move = params_move.to(self.device)
        target = target.to(self.device)

        self.optimizer_gen.zero_grad()
        gen_losses, gen_loss_total = self.calc_gen_losses(x, params_move, target)
        gen_loss_total.backward()
        self.optimizer_gen.step()

        self.optimizer_dis.zero_grad()
        dis_losses, dis_loss_total = self.calc_dis_losses(x, params_move, target)
        dis_loss_total.backward()
        self.optimizer_dis.step()

        self.log_gen_losses('Train', gen_losses, n_iter)
        self.log_dis_losses('Train', dis_losses, n_iter)

    def eval(self, batch, train_n_iter, val_n_iter):
        _, (_, x, target, params, _, _) = batch
        params_move = params[:, :self.args['params_move_count']]

        x = x.to(self.device)
        params_move = params_move.to(self.device)
        target = target.to(self.device)

        with torch.no_grad():
            gen_losses, _ = self.calc_gen_losses(x, params_move, target)
            dis_losses, _ = self.calc_dis_losses(x, params_move, target)

            self.all_val_gen_losses += [gen_losses]
            self.all_val_dis_losses += [dis_losses]

            if val_n_iter >= self.args['val_iter_count']:
                self.log_gen_losses(
                    'Validation',
                    np.stack(self.all_val_gen_losses, axis=1).mean(axis=1),
                    train_n_iter
                )
                self.log_dis_losses(
                    'Validation',
                    np.stack(self.all_val_dis_losses, axis=1).mean(axis=1),
                    train_n_iter
                )

                self.all_val_gen_losses = []
                self.all_val_dis_losses = []

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
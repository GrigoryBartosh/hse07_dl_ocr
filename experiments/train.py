import numpy as np

import torch
import torch.optim as optim

from common.config import PATH
from common.data_loading import get_loader
import common.logging as logging
from common.model import FullModelWrapper

import common.losses as losses
import common.metrics as metrics

args = {
    'load_model': False,
    'requires_grad': {
        'gen': True,
        'ocr': True
    },
    'model': {
        'gen': {
            # TODO @iisuslik43
        },
        'stacker': {
            # TODO @myutman
        },
        'ocr': {
            # TODO @grigorybartosh
        }
    },
    'train': {
        'dataset_images_name': None,
        'dataset_texts_name': None,
        'val_dataset_name': None,
        'save_iter': 10000,
        'val_iter': 100000,
        'val_iter_count': 10000,
        'batch_size': 32,
        'num_workers': 16,
        'lr': 0.001,
        'w_l2_norm': 0,
        'w_loss_gen_correct': 1.0,
        'w_loss_gen_var': 1.0,
        'w_loss_gen_detect': 10.0
    }
}


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


class Trainer():
    def __init__(self, model, args, device, summary_writer):
        self.model = model
        self.args = args
        self.device = device
        self.summary_writer = summary_writer

        self.criterion_gen_correct = losses.TextParamsCorrectLoss()
        self.criterion_gen_var = losses.TextParamsVarLoss()
        self.criterion_detect = losses.OCRLoss()

        params = filter(lambda p: p.requires_grad, model.gen.parameters())
        self.optimizer_gen = optim.Adam(
            params,
            lr=self.args['lr'],
            weight_decay=self.args['w_l2_norm']
        ) if len(params) > 0 else None

        params = filter(lambda p: p.requires_grad, model.ocr.parameters())
        self.optimizer_ocr = optim.Adam(
            params,
            lr=self.args['lr'],
            weight_decay=self.args['w_l2_norm']
        ) if len(params) > 0 else None

        self.all_val_ocr_losses = []

    def step_gen(self, x_i, x_t, n_iter):
        if self.optimizer_gen is None:
            return

        self.optimizer_gen.zero_grad()

        x_t_param = self.model.gen(x_i, x_t)

        x_mask, x_i_text_only = self.model.staсker(x_t, x_t_param)

        x_i_text = (1 - x_mask) * x_i  + x_mask * x_i_text_only

        x_res = self.model.ocr(x_i_text, x_t)

        loss_gen_correct = self.criterion_gen_correct(x_t_param)
        loss_gen_var = self.criterion_gen_var(x_t_param)

        loss_detecet = self.criterion_detect(x_res, x_t)

        loss_total = loss_gen_correct * self.args['w_loss_gen_correct'] + \
                     loss_gen_var * self.args['w_loss_gen_var'] + \
                     loss_detecet * self.args['w_loss_gen_detect']

        loss_total.backward()
        self.optimizer_gen.step()

        self.summary_writer.add_scalar(
            'Train_gen/gen_correct', loss_gen_correct.item(), n_iter)
        self.summary_writer.add_scalar(
            'Train_gen/gen_var', loss_gen_var.item(), n_iter)
        self.summary_writer.add_scalar(
            'Train_gen/detecet', loss_detecet.item(), n_iter)
        self.summary_writer.add_scalar(
            'Train_gen/total', loss_total.item(), n_iter)

    def step_ocr(self, x_i, x_t, n_iter):
        if self.optimizer_ocr is None:
            return

        self.optimizer_ocr.zero_grad()

        x_t_param = self.model.gen(x_i, x_t)

        x_mask, x_i_text_only = self.model.staсker(x_t, x_t_param)

        x_i_text = (1 - x_mask) * x_i  + x_mask * x_i_text_only

        x_res = self.model.ocr(x_i_text, x_t)

        loss_detecet = self.criterion_detect(x_res, x_t)

        loss_total = loss_detecet

        self.summary_writer.add_scalar(
            'Train_ocr/detecet', loss_detecet.item(), n_iter)
        self.summary_writer.add_scalar(
            'Train_ocr/total', loss_total.item(), n_iter)

        loss_total.backward()
        self.optimizer_ocr.step()

    def step(self, batch, n_iter):
        x_i, x_t = batch

        x_i = x_i.to(self.device)

        self.step_gen(x_i, x_t, n_iter)
        self.step_ocr(x_i, x_t, n_iter)

    def eval(self, batch, train_n_iter, val_n_iter):
        x_i, x_t = batch

        x_i = x_i.to(self.device, non_blocking=True)

        with torch.no_grad():
            x_res = self.model.ocr(x_i_text, x_t)

            loss_detecet = self.criterion_detect(x_res, x_t)

            loss_total = loss_detecet

            losses = np.array([
                loss_detecet.item(),
                loss_total.item()
            ])

        self.all_val_ocr_losses += [losses]

        if val_n_iter >= self.args['val_iter_count']:
            self.all_val_ocr_losses = np.stack(self.all_val_ocr_losses, axis=1).mean(axis=1)

            self.summary_writer.add_scalar(
                'Validation_ocr/detecet', self.all_val_ocr_losses[0], train_n_iter)
            self.summary_writer.add_scalar(
                'Validation_ocr/total', self.all_val_ocr_losses[1], train_n_iter)

            self.all_val_ocr_losses = []


if __name__ == '__main__':
    batch_size = args['train']['batch_size']
    num_workers = args['train']['num_workers']
    train_data_loader = get_loader(
        dataset_images_name=args['train']['dataset_images_name'],
        dataset_texts_name=args['train']['dataset_texts_name'],
        sset='TRAIN',
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_data_loader = get_loader(
        dataset_parallel_name=args['train']['val_dataset_name'],
        sset='VAL',
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FullModelWrapper(args['model'], load=args['load_model'])
    if args['requires_grad']['gen']:
        set_requires_grad(model.gen, True)
    if args['requires_grad']['ocr']:
        set_requires_grad(model.ocr, True)
    model.to(device)

    summary_writer = logging.get_summary_writer()

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
            model.save()
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
import multiprocessing as mp

import numpy as np

import torch
import torch.optim as optim

from common.config import PATH
from common.data_loading import get_loader
import common.logging as logging
from common.model import FullModelWrapper

import common.simple_stacker as simple_stacker

import common.losses as losses

image_size = 128
args = {
    'load_model': False,
    'requires_grad': {
        'gen': True,
        'ocr': True
    },
    'model': {
        'gen': {
            'model_image': 'resnet18',
            'model_image_pretrained': True,
            'model_text': 'resnet18',
            'model_text_pretrained': False,
            'emb_size': 256,
            'mlp_layers': [512, 128, 32, 8],
            'activ': 'relu',
            'dropout_rate': 0.2
        },
        'stacker': {
            'encoder': {
                'block': 'ResBasicBlock',
                'layers': [2, 2, 2, 2],
                'activ': 'relu'
            },
            'decoder': {
                'block': 'ResBasicBlock',
                'layers': [2, 2, 2, 2],
                'activ': 'lrelu'
            },
            'params_move_count': 5
        },
        'ocr': { # TODO
            True
        }
    },
    'train': {
        'image_size': image_size,
        'use_gen': True,
        'save_iter': 10000,
        'val_iter': 100000,
        'val_iter_count': 10000,
        'batch_size': 32,
        'num_workers': 16,
        'lr': 0.001,
        'w_l2_norm': 0,
        'log_images_count': 5,
        'parallel_process': 8
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

        self.criterion_gen_kl = losses.TextParamsKLLoss()
        self.criterion_ocr_detect = losses.OCRLoss()

        self.optimizer_gen = optim.Adam(
            filter(lambda p: p.requires_grad, model.gen.parameters()),
            lr=self.args['lr'],
            weight_decay=self.args['w_l2_norm']
        )

        self.optimizer_ocr = optim.Adam(
            filter(lambda p: p.requires_grad, model.ocr.parameters()),
            lr=self.args['lr'],
            weight_decay=self.args['w_l2_norm']
        )

        self.all_val_ocr_losses = []

    def log_gen_losses(self, phase, losses, n_iter):
        self.summary_writer.add_scalar(
            f'{phase}/KL', losses[0], n_iter)
        self.summary_writer.add_scalar(
            f'{phase}/detect', losses[1], n_iter)
        self.summary_writer.add_scalar(
            f'{phase}/total', losses[2], n_iter)

    def log_ocr_losses(self, phase, losses, n_iter):
        self.summary_writer.add_scalar(
            f'{phase}/detect', losses[0], n_iter)
        self.summary_writer.add_scalar(
            f'{phase}/total', losses[1], n_iter)

    def log_images(self, phase, x_i, x_t, n_iter):
        n = self.args['log_images_count']
        x_i, x_t = x_i[:n], x_t[:n]

        x_t_param, _, _ = self.model.gen(x_i, x_t)

        x_i_out = self.model.staсker(x_i, x_t, x_t_param)

        img_grid = torchvision.utils.make_grid(
            torch.cat((x_i, x_t[:, None, :, :], x_i_out)),
            normalize=True,
            nrow=x_i.size(0)
        )
        self.summary_writer.add_image(f'{phase}_gen', img_grid, n_iter)

    def get_bounding_boxes(texts, x_params):
        move_params = x_params.cpu().detach().numpy()[:, :5]
        n = self.args['parallel_process']
        with mp.Pool(n) as pool:
            bounding_boxes = pool.map(
                lambda t, p: simple_stacker.stack(t, self.args['image_size'], *p)[1],
                zip(texts, move_params)
            )

        bounding_boxes = bounding_boxes # TODO
        # bounding_boxes.to(self.devise)
        return bounding_boxes

    def calc_losses_gen(self, x_i, x_t, text):
        x_t_param, x_t_param_mu, x_t_param_logvar = self.model.gen(x_i, x_t)

        x_i_out = self.model.staсker(x_i, x_t, x_t_param)
        x_bb = self.get_bounding_boxes(text, x_t_param)

        x_bb_out = self.model.ocr(x_i_out)

        loss_gen_kl = self.criterion_gen_kl(x_t_param_mu, x_t_param_logvar)

        loss_ocr_detecet = self.criterion_ocr_detect(x_bb_out, x_bb)

        return loss_gen_kl, loss_ocr_detecet

    def calc_losses_no_gen(self, x_i, x_t, x_rgb, x_bb):
        x_i = self.model.staсker.stack(x_i, x_t, x_rgb)

        x_bb_out = self.model.ocr(x_bb_out, x_bb)

        loss_ocr_detecet = self.criterion_ocr_detect(x_i, x_bb)

        return loss_ocr_detecet

    def step(self, batch, n_iter):
        x_i, (texts, x_t, x_t_p, x_t_params, x_t_bb) = batch

        x_i = x_i.to(self.device)
        x_t = x_t.to(self.device)
        x_t_p = x_t_p.to(self.device)
        x_t_params = x_t_params.to(self.device)
        x_t_bb = x_t_bb # TODO

        if self.args['use_gen']:
            self.optimizer_gen.zero_grad()
            loss_gen_kl, loss_ocr_detecet = self.calc_losses_gen(x_i, x_t, text)
            loss_ocr_detecet = loss_ocr_detecet # TODO
            loss_total = loss_gen_kl + loss_ocr_detecet
            gen_losses = [loss_gen_kl.item(), loss_ocr_detecet.item(), loss_total.item()]
            loss_total.backward()
            self.optimizer_gen.step()

            self.optimizer_ocr.zero_grad()
            _, loss_ocr_detecet = self.calc_losses_gen(x_i, x_t, text)
            loss_total = loss_ocr_detecet
            ocr_losses = [loss_ocr_detecet.item(), loss_total.item()]
            loss_total.backward()
            self.optimizer_ocr.step()

            self.log_gen_losses('Train', gen_losses, n_iter)
            self.log_ocr_losses('Train', ocr_losses, n_iter)
        else:
            self.optimizer_ocr.zero_grad()
            loss_ocr_detecet = self.calc_losses_no_gen(x_i, x_t_p, x_t_params[:, 5:], x_t_bb)
            loss_total = loss_ocr_detecet
            ocr_losses = [loss_ocr_detecet.item(), loss_total.item()]
            loss_total.backward()
            self.optimizer_ocr.step()

            self.log_ocr_losses('Train', ocr_losses, n_iter)

    def eval(self, batch, train_n_iter, val_n_iter):
        x_i, (texts, x_t, x_t_p, x_t_params, x_t_bb) = batch

        x_i = x_i.to(self.device)
        x_t = x_t.to(self.device)
        x_t_p = x_t_p.to(self.device)
        x_t_params = x_t_params.to(self.device)
        x_t_bb = x_t_bb # TODO

        with torch.no_grad():
            loss_ocr_detecet = self.calc_losses_no_gen(x_i, x_t_p, x_t_params[:, 5:], x_t_bb)
            loss_total = loss_ocr_detecet
            losses = [loss_ocr_detecet.item(), loss_total.item()]

        self.all_val_ocr_losses += [losses]

        if val_n_iter >= self.args['val_iter_count']:
            self.all_val_ocr_losses = np.stack(self.all_val_ocr_losses, axis=1).mean(axis=1)

            self.log_ocr_losses(
                'Validation',
                np.stack(self.all_val_ocr_losses, axis=1).mean(axis=1),
                train_n_iter
            )

            self.all_val_ocr_losses = []

            self.log_images('Validation', x_i, x_t, target, params_move, train_n_iter)


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
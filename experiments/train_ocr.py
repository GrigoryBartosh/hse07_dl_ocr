import multiprocessing as mp

import numpy as np

import torch
import torch.optim as optim
import torchvision

from common.config import PATH
import common.utils as utils
from common.data_loading import get_loader
import common.logging as logging
from common.model import FullModelWrapper

import common.simple_stacker as simple_stacker

import common.losses as losses

image_size = 300
params_count = 8
params_move_count = 5
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
                'image_size': image_size,
                'block': 'ResBasicBlock',
                'layers': [2, 2, 2, 2],
                'activ': 'lrelu'
            },
            'params_move_count': params_move_count
        },
        'ocr': {
            'backbone': 'resnet18',
            'label_num': utils.labels_count()
        }
    },
    'train': { # TODO
        'image_size': image_size,
        'params_count': params_count,
        'params_move_count': params_move_count,
        'use_gen': False,
        'save_iter': 10000,
        'val_iter': 4,
        'val_iter_count': 4,
        'batch_size': 4, # TODO
        'num_workers': 16,
        'lr': 0.0001,
        'w_l2_norm': 0,
        'log_images_count': 5,
        'parallel_process': 8
    }
}


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def simple_stacker_by_tuple(args):
    return simple_stacker.stack(*args)


class Trainer():
    def __init__(self, model, args, device, summary_writer):
        self.model = model
        self.args = args
        self.device = device
        self.summary_writer = summary_writer

        self.box_encoder = utils.BoxEncoder()

        self.criterion_gen_kl = losses.TextParamsKLLoss()
        self.criterion_ocr_detect = losses.OCRLoss()
        self.criterion_ocr_detect.to(self.device)

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

    def log_images(self, phase, x_i, x_t, x_t_params, n_iter):
        n = self.args['log_images_count']
        x_i, x_t, x_t_params = x_i[:n], x_t[:n], x_t_params[:n]

        if self.args['use_gen']:
            x_t_params, _, _ = self.model.gen(x_i, x_t)

        x_i_out = self.model.stacker(x_i, x_t, x_t_params)

        x_bb_p, x_l_p = self.model.ocr(x_i_out)
        out = self.box_encoder.decode_batch(x_bb_p, x_l_p)

        img_bb = []
        for (bboxes, labels, scores), img in zip(out, x_i_out):
            bboxes = bboxes.cpu().numpy()
            labels = labels.cpu().numpy()
            scores = scores.cpu().numpy()
            ids = scores > 0
            bboxes, labels, scores = bboxes[ids], labels[ids], scores[ids]
            img = img.cpu().numpy()
            img = (img.transpose(1, 2, 0) + 1) / 2
            img = utils.draw_patches(img, bboxes, labels, scores=scores,
                                     order='ltrb', label_map=utils.label_to_char)
            img = (img.transpose(2, 0, 1) - 128.) / 128.
            img_bb.append(torch.tensor(img, dtype=torch.float32))
        img_bb = torch.stack(img_bb)

        x_t = x_t[:, None, :, :].expand(-1, 3, -1, -1)
        img_grid = torchvision.utils.make_grid(
            torch.cat((x_i, x_t, x_i_out, img_bb)),
            normalize=True,
            nrow=x_i.size(0)
        )
        self.summary_writer.add_image(f'{phase}_gen', img_grid, n_iter)

    def get_bboxes(self, texts, params):
        move_params = params.cpu().detach().numpy()[:, :self.args['params_move_count']]
        move_params = [(t, self.args['image_size'], *p) for t, p in zip(texts, move_params)]
        n = self.args['parallel_process']
        with mp.Pool(n) as pool:
            bboxes_labels = pool.map(simple_stacker_by_tuple, move_params)

        bboxes, labels = [], []
        for (_, bb, l) in bboxes_labels:
            bb = torch.tensor(bb, dtype=torch.float32)
            l = torch.LongTensor(l)
            bb, l = self.box_encoder.encode(bb, l)
            bboxes.append(bb.to(self.device))
            labels.append(l.to(self.device))

        bboxes = torch.stack(bboxes)
        labels = torch.stack(labels)

        return bboxes, labels

    def calc_losses_gen(self, x_i, x_t, text):
        x_t_param, x_t_param_mu, x_t_param_logvar = self.model.gen(x_i, x_t)

        x_i_out = self.model.stacker(x_i, x_t, x_t_param)
        x_bb, x_l = self.get_bboxes(text, x_t_param)

        x_bb_p, x_l_p = self.model.ocr(x_i_out)

        loss_gen_kl = self.criterion_gen_kl(x_t_param_mu, x_t_param_logvar)

        loss_ocr_detecet = self.criterion_ocr_detect(x_bb_p, x_l_p, x_bb, x_l)

        return loss_gen_kl, loss_ocr_detecet

    def calc_losses_no_gen(self, x_i, x_t, x_rgb, x_bb, x_l):
        x_i = self.model.stacker.stack(x_i, x_t, x_rgb)

        x_bb_p, x_l_p = self.model.ocr(x_i)

        loss_ocr_detecet = self.criterion_ocr_detect(x_bb_p, x_l_p, x_bb, x_l)

        return loss_ocr_detecet

    def step(self, batch, n_iter):
        x_i, (texts, x_t, x_t_p, x_t_params, x_t_bb, x_t_l) = batch

        x_i = x_i.to(self.device)
        x_t = x_t.to(self.device)
        x_t_p = x_t_p.to(self.device)
        x_t_params = x_t_params.to(self.device)
        x_t_bb = x_t_bb.to(self.device)
        x_t_l = x_t_l.to(self.device)

        if self.args['use_gen']:
            self.optimizer_gen.zero_grad()
            loss_gen_kl, loss_ocr_detecet = self.calc_losses_gen(x_i, x_t, texts)
            loss_ocr_detecet = -loss_ocr_detecet
            loss_total = loss_gen_kl + loss_ocr_detecet
            gen_losses = [loss_gen_kl.item(), loss_ocr_detecet.item(), loss_total.item()]
            loss_total.backward()
            self.optimizer_gen.step()

            self.optimizer_ocr.zero_grad()
            _, loss_ocr_detecet = self.calc_losses_gen(x_i, x_t, texts)
            loss_total = loss_ocr_detecet
            ocr_losses = [loss_ocr_detecet.item(), loss_total.item()]
            loss_total.backward()
            self.optimizer_ocr.step()

            self.log_gen_losses('Train', gen_losses, n_iter)
            self.log_ocr_losses('Train', ocr_losses, n_iter)
        else:
            self.optimizer_ocr.zero_grad()
            loss_ocr_detecet = self.calc_losses_no_gen(
                x_i,
                x_t_p,
                x_t_params[:, self.args['params_move_count']:],
                x_t_bb,
                x_t_l
            )
            loss_total = loss_ocr_detecet
            ocr_losses = [loss_ocr_detecet.item(), loss_total.item()]
            loss_total.backward()
            self.optimizer_ocr.step()

            self.log_ocr_losses('Train', ocr_losses, n_iter)

    def eval(self, batch, train_n_iter, val_n_iter):
        x_i, (texts, x_t, x_t_p, x_t_params, x_t_bb, x_t_l) = batch

        x_i = x_i.to(self.device)
        x_t = x_t.to(self.device)
        x_t_p = x_t_p.to(self.device)
        x_t_params = x_t_params.to(self.device)
        x_t_bb = x_t_bb.to(self.device)
        x_t_l = x_t_l.to(self.device)

        with torch.no_grad():
            loss_ocr_detecet = self.calc_losses_no_gen(
                x_i,
                x_t_p,
                x_t_params[:, self.args['params_move_count']:],
                x_t_bb,
                x_t_l
            )
            loss_total = loss_ocr_detecet
            losses = [loss_ocr_detecet.item(), loss_total.item()]

            self.all_val_ocr_losses += [losses]

            if val_n_iter >= self.args['val_iter_count']:
                self.log_ocr_losses(
                    'Validation',
                    np.stack(self.all_val_ocr_losses, axis=1).mean(axis=1),
                    train_n_iter
                )

                self.all_val_ocr_losses = []

                self.log_images('Validation', x_i, x_t, x_t_params, train_n_iter)


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
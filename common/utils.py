import os
import json
import shutil
import itertools

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.config import PATH

__all__ = ['make_dir', 'remove_dir', 'activation_by_name']


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remove_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def write_arguments(summary_writer, args):
    s = json.dumps(args, indent=4)
    s = s.replace(' ', '&nbsp;')
    s = s.replace('\n', '  \n')
    summary_writer.add_text('args', s, )


def activation_by_name(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'selu':
        return nn.SELU(inplace=True)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'none':
        return None
    else:
        assert False, f"Unsupported activation: {activation}"


def char_to_label(c):
    if c >= 'A' and c <= 'Z':
        return ord(c) - ord('A') + 1
    if c >= 'a' and c <= 'z':
        return 26 + ord(c) - ord('a') + 1
    if c >= '0' and c <= '9':
        return 52 + ord(c) - ord('0') + 1
    return None


def label_to_char(c):
    if c >= 1 and c <= 26:
        return chr(c - 1 + ord('A'))    
    if c >= 27 and c <= 52:
        return chr(c - 27 + ord('a'))
    if c >= 53 and c <= 62:
        return chr(c - 53 + ord('0'))


def is_char_valid(c):
    return char_to_label(c) is not None


def labels_count():
    return 63


class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios,
                 scale_xy=0.1, scale_wh=0.2):

        self.feat_size = feat_size
        self.fig_size = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        self.steps = steps
        self.scales = scales

        fk = fig_size / np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        for idx, sfeat in enumerate(self.feat_size):

            sk1 = scales[idx] / fig_size
            sk2 = scales[idx + 1] / fig_size
            sk3 = np.sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1 * np.sqrt(alpha), sk1 / np.sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.tensor(self.default_boxes)
        self.dboxes.clamp_(min=0, max=1)
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.dboxes_ltrb
        if order == "xywh": return self.dboxes


def get_dboxes():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


def calc_iou_tensor(box1, box2):
    N = box1.size(0)
    M = box2.size(0)

    be1 = box1.unsqueeze(1).expand(-1, M, -1)
    be2 = box2.unsqueeze(0).expand(N, -1, -1)

    lt = torch.max(be1[:,:,:2], be2[:,:,:2])
    rb = torch.min(be1[:,:,2:], be2[:,:,2:])

    delta = rb - lt
    delta[delta < 0] = 0
    intersect = delta[:,:,0] * delta[:,:,1]

    delta1 = be1[:,:,2:] - be1[:,:,:2]
    area1 = delta1[:,:,0] * delta1[:,:,1]
    delta2 = be2[:,:,2:] - be2[:,:,:2]
    area2 = delta2[:,:,0]  *delta2[:,:,1]

    iou = intersect / (area1 + area2 - intersect)
    return iou


class BoxEncoder():
    def __init__(self):
        dboxes = get_dboxes()
        self.dboxes = dboxes(order="ltrb")
        self.dboxes_xywh = dboxes(order="xywh").unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0)
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, bboxes_in, labels_in, criteria=0.5):
        ious = calc_iou_tensor(bboxes_in, self.dboxes)
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)

        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        masks = best_dbox_ious > criteria
        labels_out = torch.zeros(self.nboxes, dtype=torch.long)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]
        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]
        x, y, w, h = 0.5 * (bboxes_out[:, 0] + bboxes_out[:, 2]), \
                     0.5 * (bboxes_out[:, 1] + bboxes_out[:, 3]), \
                     -bboxes_out[:, 0] + bboxes_out[:, 2], \
                     -bboxes_out[:, 1] + bboxes_out[:, 3]
        bboxes_out[:, 0] = x
        bboxes_out[:, 1] = y
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h
        return bboxes_out.permute(1, 0), labels_out

    def scale_back_batch(self, bboxes_in, scores_in):
        if bboxes_in.device == torch.device("cpu"):
            self.dboxes = self.dboxes.cpu()
            self.dboxes_xywh = self.dboxes_xywh.cpu()
        else:
            self.dboxes = self.dboxes.cuda()
            self.dboxes_xywh = self.dboxes_xywh.cuda()

        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]

        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        l, t, r, b = bboxes_in[:, :, 0] - 0.5*bboxes_in[:, :, 2],\
                     bboxes_in[:, :, 1] - 0.5*bboxes_in[:, :, 3],\
                     bboxes_in[:, :, 0] + 0.5*bboxes_in[:, :, 2],\
                     bboxes_in[:, :, 1] + 0.5*bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l
        bboxes_in[:, :, 1] = t
        bboxes_in[:, :, 2] = r
        bboxes_in[:, :, 3] = b

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_batch(self, bboxes_in, scores_in, criteria=0.45, max_output=200):
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

        output = []
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            output.append(self.decode_single(bbox, prob, criteria, max_output))
        return output

    def decode_single(self, bboxes_in, scores_in, criteria, max_output, max_num=200):
        bboxes_out = []
        scores_out = []
        labels_out = []

        for i, score in enumerate(scores_in.split(1, 1)):
            if i == 0: continue

            score = score.squeeze(1)
            mask = score > 0.05

            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0: continue

            score_sorted, score_idx_sorted = score.sort(dim=0)

            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []

            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                bboxes_sorted = bboxes[score_idx_sorted, :]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                iou_sorted = calc_iou_tensor(bboxes_sorted, bboxes_idx).squeeze()
                score_idx_sorted = score_idx_sorted[iou_sorted < criteria]
                candidates.append(idx)

            bboxes_out.append(bboxes[candidates, :])
            scores_out.append(score[candidates])
            labels_out.extend([i] * len(candidates))

        if not bboxes_out:
            return [torch.tensor([]) for _ in range(3)]

        bboxes_out, labels_out, scores_out = torch.cat(bboxes_out, dim=0), \
               torch.tensor(labels_out, dtype=torch.long), \
               torch.cat(scores_out, dim=0)


        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]


def draw_patches(img, bboxes, labels, scores=None, order="xywh", label_map={}):
    im = Image.fromarray(img.astype(np.uint8))
    if len(bboxes) != 0:   
        draw = ImageDraw.Draw(im)     
        if order == "ltrb":
            x0, y0, x1, y1 = bboxes[:, 0],  bboxes[:, 1],  bboxes[:, 2],  bboxes[:, 3]
        else:
            cx, cy, w, h = bboxes[:, 0],  bboxes[:, 1],  bboxes[:, 2],  bboxes[:, 3]
            x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

        htot, wtot, _ = imm.shape
        x0 *= wtot
        y0 *= htot
        x1 *= wtot
        y1 *= htot
        x1 += 1
        y1 += 1

        bboxes = zip(x0, y0, x1, y1)
        scores = scores if scores is not None else [None] * len(labels)
        for (x0, y0, x1, y1), label, score in zip(bboxes, labels, scores):
            if label == 0:
                continue
            if label_map != {}:
                label = label_map[label]
            else:
                label = label_to_char(label)
            font = ImageFont.truetype(PATH['FONTS']['LOBSTER'], int(x1 - x0))
            delta = font.getsize(label)[1] + 5
            draw.rectangle([x0, y0, x1, y1], outline='#ff0000', width=int(htot / 200 + 1))
            draw.text([x0, y0 - delta], label, fill='#ff00ff', font=font)
    return np.array(im)

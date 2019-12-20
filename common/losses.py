import torch
import torch.nn as nn

import common.utils as utils

__all__ = [
    'StackerLoss',
    'TextParamsKLLoss',
    'OCRLoss'
]


class StackerLoss(nn.Module):
    def __init__(self):
        super(StackerLoss, self).__init__()

    def forward(self, x, target):
        loss = (x - target) ** 2
        target = (target + 1) / 2
        loss = loss.sum((1, 2)) / (target ).sum((1, 2))
        return loss.mean()


class TextParamsKLLoss(nn.Module):
    def __init__(self):
        super(TextParamsKLLoss, self).__init__()

    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())


class OCRLoss(nn.Module):
    def __init__(self):
        super(OCRLoss, self).__init__()

        dboxes = utils.get_dboxes()

        self.scale_xy = 1.0 / dboxes.scale_xy
        self.scale_wh = 1.0 / dboxes.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.dboxes = nn.Parameter(
            dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0),
            requires_grad=False
        )
        self.con_loss = nn.CrossEntropyLoss(reduction='none')

    def _loc_vec(self, loc):
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, ]
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        mask = glabel > 0
        pos_num = mask.sum(dim=1)

        vec_gd = self._loc_vec(gloc)

        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
        sl1 = (mask.float() * sl1).sum(dim=1)

        con = self.con_loss(plabel, glabel)

        con_neg = con.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        closs = (con * (mask.float() + neg_mask.float())).sum(dim=1)

        total_loss = sl1 + closs
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss * num_mask / pos_num).mean(dim=0)
        return ret

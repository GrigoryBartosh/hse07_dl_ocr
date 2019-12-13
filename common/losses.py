import torch
import torch.nn as nn

__all__ = [
    'StackerLoss',
    'TextParamsKLLoss',
    'OCRLoss'
]


class StackerLoss(nn.Module):
    def __init__(self):
        super(StackerLoss, self).__init__()

    def forward(self, x, target):
        return ((x - target) ** 2).mean()


class TextParamsKLLoss(nn.Module):
    def __init__(self):
        super(TextParamsKLLoss, self).__init__()

    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())


class OCRLoss(nn.Module): # TODO
    def __init__(self):
        super(OCRLoss, self).__init__()

    def forward(self, x_res, x_t):
        pass
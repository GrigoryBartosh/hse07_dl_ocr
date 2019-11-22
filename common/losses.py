import torch
import torch.nn as nn

__all__ = [
    'TextParamsCorrectLoss', 'TextParamsVarLoss', 
    'OCRLoss'
]


# TODO @iisuslik43
class TextParamsCorrectLoss(nn.Module):
    def __init__(self):
        super(TextParamsCorrectLoss, self).__init__()

    def forward(self, x_t_param):
        pass


# TODO @iisuslik43
class TextParamsVarLoss(nn.Module):
    def __init__(self):
        super(TextParamsVarLoss, self).__init__()

    def forward(self, x_t_param):
        pass


# TODO @grigorybartosh
class OCRLoss(nn.Module):
    def __init__(self):
        super(OCRLoss, self).__init__()

    def forward(self, x_res, x_t):
        pass
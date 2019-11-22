import os
import datetime

from torch.utils.tensorboard import SummaryWriter

from common.config import PATH
import common.utils as utils

__all__ = ['get_summary_writer']


def get_summary_writer():
    name = str(datetime.datetime.now())[:19]
    utils.make_dir(PATH['TF_LOGS'])
    logs_path = os.path.join(PATH['TF_LOGS'], name)
    return SummaryWriter(logs_path)

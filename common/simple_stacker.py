import os

import numpy as np

__all__ = ['stack']


def stack(text, image_size,
          rotate_angle=0, ratio=0, scale=0,
          shift_x=0, shift_y=0):
    return np.zeros((image_size, image_size)), []
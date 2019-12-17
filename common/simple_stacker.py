import matplotlib.pyplot as plt

import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from torchvision.transforms.functional import center_crop

from common.config import PATH
import common.utils as utils

BASIC_FONT_SIZE = 32
FONT_PATH = PATH['FONTS']['LOBSTER']
BASIC_FONT = ImageFont.truetype(FONT_PATH, BASIC_FONT_SIZE)
MODE = 'L'
MAX_RATIO = 2
MIN_SCALE = 12
WHITE_PIXEL = 255
RED_PIXEL = 127

__all__ = ['stack']


def create_image_of_size(x, y):
    return Image.new(MODE, (x, y))


def bounding_box(c, angle, ratio):
    times = 8
    
    x, y = BASIC_FONT.getsize(c)
    _, dy = BASIC_FONT.getoffset(c)
    y -= dy

    pillowImage = create_image_of_size(times * BASIC_FONT_SIZE, int(times * BASIC_FONT_SIZE / ratio))
    ImageDraw.Draw(pillowImage).text(
        (
            times * BASIC_FONT_SIZE / 2. - x / 2,
            times * BASIC_FONT_SIZE / 2. / ratio - y / 2 - dy
        ), c, WHITE_PIXEL, font=BASIC_FONT
    )

    pillowImage = pillowImage.resize((times * BASIC_FONT_SIZE, times * BASIC_FONT_SIZE))
    
    pillowImage = pillowImage.rotate(angle)
    pillowImage = center_crop(pillowImage, times * BASIC_FONT_SIZE // 2)
    arr = np.asarray(pillowImage)
    
    ids = np.where((arr > 0))
    x1 = ids[1].min()
    x2 = ids[1].max()
    y1 = ids[0].min()
    y2 = ids[0].max()
    
    return np.array([x1, y1, x2, y2], dtype=np.float64) / BASIC_FONT_SIZE - times / 4.
    

def rotate_tuple(x, y, a):
    return (x * np.cos(a) + y * np.sin(a), -x * np.sin(a) + y * np.cos(a))

def rotated_rect(x, y, angle_rad):
    return (
        x * abs(np.cos(angle_rad)) + y * abs(np.sin(angle_rad)),
        y * abs(np.cos(angle_rad)) + x * abs(np.sin(angle_rad))
    )

def build_bb(text, size, x, y, angle, ratio, font1, l, shift_x, shift_y):
    angle_rad = angle * np.pi
    angle_deg = angle * 180.
    
    ans = []
    label = []
    
    _, offset = font1.getoffset(text)

    bboxes = {}
    for i in range(len(text)):
        if text[i].isspace():
            continue

        c = text[i]

        dx, _ = font1.getsize(text[:i + 1])
        dx1, dy1 = font1.getsize(c)
        _, offset1 = font1.getoffset(c)
        dy1 -= offset1
        
        center_x = -x / 2. + dx - dx1 / 2
        center_y = -y / 2 - offset + offset1 + dy1 / 2

        center_y *= ratio

        center_x += shift_x
        center_y += shift_y
        
        center_x, center_y = rotate_tuple(center_x, center_y, angle_rad)
        
        if c in bboxes:
            bbox = bboxes[c]
        else:
            bbox = bounding_box(c, angle_deg, ratio)
            bboxes[c] = bbox

        x1, y1, x2, y2 = bbox * l
        x1 += center_x + size / 2.
        x2 += center_x + size / 2.
        y1 += center_y + size / 2.
        y2 += center_y + size / 2.
        
        ans.append(np.array([x1, y1, x2, y2]) / size)
        label.append(utils.char_to_label(text[i]))
    
    return ans, label

def stack(text, size, angle=0, ratio=0, scale=0, shift_x=0, shift_y=0):
    angle_rad = angle * np.pi
    angle_deg = angle * 180.

    ratio = MAX_RATIO ** ratio

    scale = (scale + 1) / 2.

    x, y = BASIC_FONT.getsize(text)
    _, dy = BASIC_FONT.getoffset(text)
    y -= dy

    l_max = BASIC_FONT_SIZE * size / max(*rotated_rect(x, y * ratio, angle_rad))
    l = scale * (l_max - MIN_SCALE) + MIN_SCALE
    
    font1 = ImageFont.truetype(FONT_PATH, int(l))
    x1, y1 = font1.getsize(text)
    _, dy1 = font1.getoffset(text)
    y1 -= dy1
    
    w, h = rotated_rect(x1, y1 * ratio, angle_rad)
    
    shift_x *= (size - w) / 2
    shift_y *= (size - h) / 2
    shift_x, shift_y = rotate_tuple(shift_x, shift_y, -angle_rad)

    start_x = size - x1 // 2 + shift_x
    start_y = size / ratio - y1 // 2 - dy1 + shift_y / ratio

    pillowImage = create_image_of_size(2 * size, int(2 * size / ratio))
    ImageDraw.Draw(pillowImage, MODE).text((start_x, start_y), text, WHITE_PIXEL, font=font1)
    
    pillowImage = pillowImage.resize((2 * size, 2 * size))
    pillowImage = pillowImage.rotate(angle_deg)
    pillowImage = center_crop(pillowImage, size)
    
    bbs, label = build_bb(text, size, x1, y1, angle, ratio, font1, l, shift_x, shift_y)
    
    return (np.asarray(pillowImage) * 2. / 255. - 1.), bbs, label
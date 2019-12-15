import matplotlib.pyplot as plt

import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from torchvision.transforms.functional import center_crop

from common.config import PATH
import common.utils as utils

WHITE_PIXEL = 255
RED_PIXEL = 127
BASIC_FONT_SIZE = 32
FONT_PATH = PATH['FONTS']['LOBSTER']
BASIC_FONT = ImageFont.truetype(FONT_PATH, BASIC_FONT_SIZE)
MAX_RATIO = 3
MODE = 'L'

__all__ = ['stack']


def create_image_of_size(x, y):
    return Image.new(MODE, (y, x))

def is_not_black(pixel):
    return pixel > 0

def bounding_box(c, angle, ratio):
    times = 8
    
    pillowImage = create_image_of_size(int(times * BASIC_FONT_SIZE * ratio), times * BASIC_FONT_SIZE)
    x, y = BASIC_FONT.getsize(c)
    ImageDraw.Draw(pillowImage).text(
        (
            (times / 2.) * BASIC_FONT_SIZE - x / 2,
            (times / 2.) * BASIC_FONT_SIZE * ratio - y / 2
        ), c, WHITE_PIXEL, font=BASIC_FONT
    )
    pillowImage = pillowImage.resize((times * BASIC_FONT_SIZE, times * BASIC_FONT_SIZE))
    pillowImage = pillowImage.rotate(angle)
    pillowImage = center_crop(pillowImage, times / 2 * BASIC_FONT_SIZE)
    arr = np.asarray(pillowImage)
    
    x1, y1, x2, y2 = [times * BASIC_FONT_SIZE, times * BASIC_FONT_SIZE, -1, -1]
    for i in range((times // 2) * BASIC_FONT_SIZE):
        for j in range((times // 2) * BASIC_FONT_SIZE):
            if is_not_black(arr[i][j]):
                x1 = min(x1, j)
                y1 = min(y1, i)
                x2 = max(x2, j)
                y2 = max(y2, i)
    
    draw = ImageDraw.Draw(pillowImage)
    draw.rectangle([x1, y1, x2, y2], outline=RED_PIXEL)
    draw.rectangle([
    	BASIC_FONT_SIZE * times / 4. - x / 2,
    	BASIC_FONT_SIZE * times / 4. - y / 2,
    	BASIC_FONT_SIZE * times / 4. + x / 2,
    	BASIC_FONT_SIZE * times / 4. + y / 2
    ])
    
    return pillowImage, np.array([x1, y1, x2, y2], dtype=np.float64) / BASIC_FONT_SIZE - times / 4.
    

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
    
    for i in range(len(text)):
        if text[i].isspace():
            continue
        dx, _ = font1.getsize(text[:i + 1])
        dx1, dy1 = font1.getsize(text[i])
        dx -= dx1 / 2
        
        dy = ((dy1 / 2.) - (y / 2.)) / ratio
        
        center_x = dx - x / 2. + shift_x
        center_y = shift_y + dy
        
        x0, y0 = rotate_tuple(center_x, center_y, angle_rad)
        
        _, v = bounding_box(text[i], angle_deg, ratio)
        x1, y1, x2, y2 = v * l
        x1 += x0 + size / 2.
        x2 += x0 + size / 2.
        y1 += y0 + size / 2.
        y2 += y0 + size / 2.
        
        ans.append(np.array([x1, y1, x2, y2]) / size)
        label.append(utils.char_to_label(text[i]))
    
    return ans, label

def stack(text, size, angle=0, ratio=0, scale=0, shift_x=0, shift_y=0):
    ratio = MAX_RATIO ** ratio
    
    angle_rad = angle * np.pi
    angle_deg = angle * 180.
    
    pillowImage = create_image_of_size(int(2 * size * ratio), 2 * size)

    draw = ImageDraw.Draw(pillowImage, MODE)

    x, y = BASIC_FONT.getsize(text)
    l = BASIC_FONT_SIZE * scale * size / max(*rotated_rect(x, y * ratio, angle_rad))
    
    font1 = ImageFont.truetype(FONT_PATH, int(l))
    x1, y1 = font1.getsize(text)
    
    w, h = rotated_rect(x1, y1 * ratio, angle_rad)
    
    shift_x *= (size - w) / 2
    shift_y *= (size - h) / 2
    shift_x, shift_y = rotate_tuple(shift_x, shift_y, -angle_rad)

    start_x = size - x1 / 2 + shift_x
    start_y = size * ratio - y1 / 2 + shift_y * ratio
    draw.text((start_x, start_y), text, WHITE_PIXEL, font=font1)
    pillowImage = pillowImage.resize((2 * size, 2 * size))
    pillowImage = pillowImage.rotate(angle_deg)
    pillowImage = center_crop(pillowImage, size)
    
    bbs, label = build_bb(text, size, x1, y1, angle, ratio, font1, l, shift_x, shift_y)
    
    return (np.asarray(pillowImage) - 128) / 128., bbs, label
import matplotlib.pyplot as plt

import numpy as np

import common.utils as utils
import common.simple_stacker as simple_stacker


if __name__ == '__main__':
    img, bboxes, labels = simple_stacker.stack('hello world', 300)

    img = (img + 1) / 2
    img = np.expand_dims(img, axis=2)
    img = np.concatenate((img, img, img), axis=2)

    img = utils.draw_patches(img, bboxes, labels, order='ltrb', label_map=utils.label_to_char)

    plt.imshow(img)
    plt.show()
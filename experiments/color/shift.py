#!/usr/bin/env python3

import numpy as np
from matplotlib import image, pyplot

import src.color as color


def experiment():
    trees = "resource/smaller.jpg"
    trees = np.array(image.imread(trees), dtype=float)

    np.apply_along_axis(color.shift(0.8, 0.5, 1.2), 2, trees)

    trees = trees.astype(int)

    # print(trees)
    pyplot.imshow(trees)
    pyplot.show()


if __name__ == "__main__":
    experiment()

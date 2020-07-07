#!/usr/bin/env python3

import numpy as np
from matplotlib import image, pyplot


def shift(red: int, green: int, blue: int):
    scalars = np.array([red, green, blue])

    def func(rgb: np.array):
        rgb[:] = np.multiply(rgb, scalars)

    return func


def experiment():
    trees = "resource/smaller.jpg"
    trees = np.array(image.imread(trees), dtype=float)

    np.apply_along_axis(shift(0.8, 0.5, 1.2), 2, trees)

    trees = trees.astype(int)

    # print(trees)
    pyplot.imshow(trees)
    pyplot.show()


if __name__ == "__main__":
    experiment()

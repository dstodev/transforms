import numpy as np


def shear(scale_x, scale_y):
    return np.array([
        [1, scale_x],
        [scale_y, 1]
    ])

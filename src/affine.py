import numpy as np

from src.utility import transform


def shear(scale_x, scale_y):
    A = np.array([
        [1, scale_x],
        [scale_y, 1]
    ])

    return transform(A)

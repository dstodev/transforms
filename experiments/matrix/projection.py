#!/usr/bin/env python3

import numpy as np
from matplotlib import image, patches, pyplot
from numpy.linalg import norm

import src.style as style
import src.utility as utility
from src.squarevariabletransform import SquareVariableTransform


def experiment():
    _fig, ax = pyplot.subplots(figsize=(5, 5))

    pyplot.grid(alpha=0.15, linestyle="--")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)

    # Set up patch vertices
    origin_x = 0.5
    origin_y = 0.5
    gray = utility.square(origin_x, origin_y)
    red = utility.square(origin_x, origin_y)
    red = np.hstack((red, np.zeros((red.shape[0], 1), dtype=red.dtype)))
    red = np.hstack((red, np.ones((red.shape[0], 1), dtype=red.dtype)))

    svt = SquareVariableTransform((0.5, 0.5), 1)
    svt._indices[(1, 2)] = None
    svt._indices[(3, 2)] = None
    svt._indices[(1, 1)] = None
    svt._indices[(4, 1)] = None
    svt._get_transform_matrix()

    # Transform patches
    P = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    red = utility.apply_transform(P, red)
    red = utility.from_homogenous(red)

    # Add patches
    ax.add_patch(patches.Polygon(gray, **style.gray))
    ax.add_patch(patches.Polygon(red, **style.red))

    pyplot.show()


if __name__ == "__main__":
    experiment()

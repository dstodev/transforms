#!/usr/bin/env python3

import numpy as np
from matplotlib import image, patches, pyplot

import src.affine as affine
import src.style as style
import src.utility as utility


"""Commutativity experiments

Notation
    x  -> column vector
    x' -> row vector (transposed column vector)
    A  -> matrix

Properties
    (A')' = A
    (AB)' = B'A'

Questions
    Is xAB = BAx ?
    No:
        xAB = B'A'x'
"""


def experiment():
    """Show that matrix multiplication (dot product) is not commutative, but that the order of operations can be
    reversed using matrix transposition.
    """
    _fig, ax = pyplot.subplots(figsize=(5, 5))

    pyplot.grid(alpha=0.15, linestyle="--")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)

    # Set up patch vertices
    origin_x = 0.5
    origin_y = 0.5
    gray = utility.square(origin_x, origin_y)
    red = utility.square(origin_x, origin_y)
    green = utility.square(origin_x, origin_y)
    blue = utility.square(origin_x, origin_y)
    purple = utility.square(origin_x, origin_y)

    # Transform patches
    X = np.array([  # Horizontal shear
        [1, 0.5],
        [0, 1]
    ])
    Y = np.array([  # Vertical shear
        [1, 0],
        [0.5, 1]
    ])
    T = np.dot(Y, X)  # YXv -> Shear horizontally, then vertically
    T_ = np.dot(X.T, Y.T)  # YXv --> v'X'Y'

    # Shear horizontally, then vertically.
    np.apply_along_axis(affine.shear(0.5, 0), 1, blue)
    np.apply_along_axis(affine.shear(0, 0.5), 1, blue)

    np.apply_along_axis(utility.transform(T), 1, red)  # YXv
    np.apply_along_axis(utility.transform(T_, row_vector=True), 1, green)  # v'X'Y'
    np.apply_along_axis(utility.transform(T.transpose(), row_vector=True), 1, purple)  # v'(YX)'

    # Add patches
    ax.add_patch(patches.Polygon(gray, **style.gray))
    ax.add_patch(patches.Polygon(red, **style.red))
    ax.add_patch(patches.Polygon(green, **style.green))
    ax.add_patch(patches.Polygon(blue, **style.blue))
    ax.add_patch(patches.Polygon(purple, **style.purple))

    pyplot.show()


if __name__ == "__main__":
    experiment()

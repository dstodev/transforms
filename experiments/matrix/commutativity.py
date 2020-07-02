#!/usr/bin/env python3

import numpy as np
from matplotlib import image, patches, pyplot, widgets

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
    fig: pyplot.Figure = pyplot.figure(figsize=(5, 6))
    ax: pyplot.Axes = fig.add_axes([0.1, 0.2, 0.85, 0.71])

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

    red = utility.apply_transform(T, red)  # YXv
    green = utility.apply_transform(T_, green, row_vector=True)  # v'X'Y'
    purple = utility.apply_transform(T.transpose(), purple, row_vector=True)  # v'(YX)'

    # Add patches
    ax.add_patch(patches.Polygon(gray, **style.gray))
    ax.add_patch(patches.Polygon(red, **style.red))
    ax.add_patch(patches.Polygon(green, **style.green))
    blue_handle: patches.Patch = ax.add_patch(patches.Polygon(blue, **style.blue))
    ax.add_patch(patches.Polygon(purple, **style.purple))

    def update_blue(scale):
        # Shear horizontally, then vertically.
        blue = utility.square(origin_x, origin_y)
        blue = utility.apply_transform(affine.shear(scale, 0), blue)
        blue = utility.apply_transform(affine.shear(0, scale), blue)
        blue_handle.set_xy(blue)

    shear_ax_blue = fig.add_axes([0.1, 0.05, 0.8, 0.025])
    ax.add_child_axes(shear_ax_blue)

    shear_blue = widgets.Slider(shear_ax_blue, "Shear Scale", 0, 1, 0.5)
    shear_blue.on_changed(update_blue)
    update_blue(0.5)

    pyplot.show()


if __name__ == "__main__":
    experiment()

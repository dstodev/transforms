#!/usr/bin/env python3

import numpy as np
from matplotlib import image, patches, pyplot, widgets
from numpy.linalg import norm

import src.style as style
import src.utility as utility
from src.interactivesquare import InteractiveSquare


def experiment():
    fig: pyplot.Figure = pyplot.figure(figsize=(5, 6))
    ax: pyplot.Axes = fig.add_axes([0.1, 0.2, 0.85, 0.71])

    pyplot.grid(alpha=0.15, linestyle="--")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)

    # Set up patch vertices
    origin_x = 0.5
    origin_y = 0.5
    gray = utility.square(origin_x, origin_y)
    red = utility.square(origin_x, origin_y, add_coords=[0, 1])

    T = np.array([
        [1, 0],
        [0.5, 1]
    ])

    sq = InteractiveSquare((0.5, 0.5), 1, style=style.purple, transform_matrix=T)
    ax.add_patch(sq.get_patch())

    shear_ax_blue = fig.add_axes([0.1, 0.05, 0.8, 0.025])
    ax.add_child_axes(shear_ax_blue)

    shear_blue = widgets.Slider(shear_ax_blue, "Shear Scale", 0, 1, 0.5)

    sq.register_slider((0, 1), shear_blue)

    # # Transform patches
    # P = np.array([
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 0]
    # ])

    # red = utility.apply_transform(P, red)
    # red = utility.from_homogenous(red)

    # # Add patches
    # ax.add_patch(patches.Polygon(gray, **style.gray))
    # ax.add_patch(patches.Polygon(red, **style.red))

    pyplot.show()


if __name__ == "__main__":
    experiment()

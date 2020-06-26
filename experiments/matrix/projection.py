#!/usr/bin/env python3

import numpy as np
from matplotlib import image, patches, pyplot

import src.style as style
import src.utility as utility


def experiment():
    _fig, ax = pyplot.subplots(figsize=(5, 5))

    pyplot.grid(alpha=0.15, linestyle="--")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)

    # Set up patch vertices
    origin_x = 0.5
    origin_y = 0.5
    gray = utility.square(origin_x, origin_y)
    red = utility.square(origin_x, origin_y, homogenous=True)

    # Transform patches

    # Add patches
    ax.add_patch(patches.Polygon(gray, **style.gray))
    ax.add_patch(patches.Polygon(red, **style.red))

    pyplot.show()


if __name__ == "__main__":
    experiment()

#!/usr/bin/env python3

import numpy as np
from matplotlib import gridspec, image, patches, pyplot, ticker, widgets
from numpy.linalg import norm

import src.style as style
import src.utility as utility
from src.interactivesquare import InteractiveSquare


def experiment():
    fig: pyplot.Figure = pyplot.figure()
    grid = gridspec.GridSpec(3, 1, figure=fig)
    ax: pyplot.Axes = pyplot.subplot(grid[:2, :])

    num_sliders = 5
    sliders = gridspec.GridSpecFromSubplotSpec(num_sliders, 1, grid[2, :])

    ax.axis("equal")
    ax.grid(alpha=0.15, linestyle="--")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

    # Set up patches
    origin = (0.5, 0.5)
    gray = InteractiveSquare(origin, style=style.gray)
    ax.add_patch(gray.get_patch())
    # red = InteractiveSquare(origin, add_coords=[0, 1], style=style.red)
    # ax.add_patch(red.get_patch())

    T = np.array([
        [1,   0],
        [0.5, 1]
    ])

    sq = InteractiveSquare((0.5, 0.5), 1, style=style.purple)
    ax.add_patch(sq.get_patch())

    ax_shear_x = pyplot.subplot(sliders[0, 0])
    ax_shear_y = pyplot.subplot(sliders[1, 0])
    ax.add_child_axes(ax_shear_x)

    shear_x = widgets.Slider(ax_shear_x, "Shear Scale X", 0, 1, 0.5)
    shear_y = widgets.Slider(ax_shear_y, "Shear Scale Y", 0, 1, 0.5)
    sq.register_slider(0, (0, 1), shear_x)
    sq.register_slider(0, (1, 0), shear_y)

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

    ax.relim()
    ax.autoscale_view()

    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":
    experiment()

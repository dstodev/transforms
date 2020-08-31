#!/usr/bin/env python3

import math

import numpy as np
from matplotlib import gridspec, image, patches, pyplot, ticker, widgets
from numpy.linalg import norm

import src.style as style
import src.utility as utility
from src.interactivesquare import InteractiveSquare
from src.mutablematrix import MutableMatrix
from src.sequence import Sequence


def experiment():
    viewport_ratio = 4  # 3 rows/4 rows for viewport, 1 row/4 rows for sliders
    num_sliders = 4

    figure: pyplot.Figure = pyplot.figure()
    grid: pyplot.GridSpec = gridspec.GridSpec(viewport_ratio, 1, figure=figure)
    axes: pyplot.Axes = pyplot.subplot(grid[:viewport_ratio - 1, :])

    sliders = gridspec.GridSpecFromSubplotSpec(num_sliders, 1, grid[-1, :])

    # Configure style
    axes.axis("equal")
    axes.grid(alpha=0.15, linestyle="--")
    axes.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axes.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

    # Set up gray (baseline) patch
    origin = (0, 0)
    gray = InteractiveSquare(axes, origin, style=style.gray)
    axes.add_patch(gray.get_patch())

    # Set up green (interactive) patch
    K = np.array([  # Intrinsic paremeter matrix
        [1, 0, 0, 0],  # [αₓ γ  μ₀ 0]
        [0, 1, 0, 0],  # [0  αᵧ ν₀ 0]
        [0, 0, 1, 0]   # [0  0  1  0]
    ])

    Rx = np.identity(3)  # Extrinsic parameter matrix
    Ry = np.identity(3)  # [R,3x3 T,3x1]
    Rz = np.identity(3)  # [0,1x3     1]
    T = np.array([[0, 0, 0]])
    B = np.array([[0, 0, 0, 1]])

    green = InteractiveSquare(axes, (0, 0), 1, style=style.green, convert_2d=utility.from_homogenous, label_vertices=True)
    axes.add_patch(green.get_patch())

    # TODO: Possible to render square differently, so that we can see if the square is "upside down".
    #       Maybe put text next to each vertex?

    green.register_transform(K)
    # Rx, Ry, Rz, T, B must be resolved before K can dot it.
    RT = Sequence()
    RT.register_node(MutableMatrix(Rx), None)
    RT.register_node(MutableMatrix(Ry), np.dot)
    RT.register_node(MutableMatrix(Rz), np.dot)
    RT.register_node(MutableMatrix(T), lambda a, b: np.concatenate((a, b.T), axis=1))
    RT.register_node(MutableMatrix(B), lambda a, b: np.concatenate((a, b), axis=0))
    green.register_transform(RT)

    # TODO: Do not register sliders to x, y, z angles. Register them to pitch, yaw, and roll.
    #       Would need to register matrices/sliders with functions to calculate world rotations so that
    #       pitch, yaw, roll are valid. Need to convert from tait-bryan (intrinsic) to Euler (extrinsic) angles.
    slider_1 = widgets.Slider(pyplot.subplot(sliders[0, 0]), "Rotate: X", 0, 360, 0, **style.darkgreen)
    green.register_slider((1, 0), (1, 1), slider_1, lambda v: math.cos(math.radians(v)))
    green.register_slider((1, 0), (1, 2), slider_1, lambda v: (-1 * math.sin(math.radians(v))))
    green.register_slider((1, 0), (2, 1), slider_1, lambda v: math.sin(math.radians(v)))
    green.register_slider((1, 0), (2, 2), slider_1, lambda v: math.cos(math.radians(v)))

    slider_2 = widgets.Slider(pyplot.subplot(sliders[1, 0]), "Rotate: Y", 0, 360, 0, **style.darkgreen)
    green.register_slider((1, 1), (0, 0), slider_2, lambda v: math.cos(math.radians(v)))
    green.register_slider((1, 1), (0, 2), slider_2, lambda v: math.sin(math.radians(v)))
    green.register_slider((1, 1), (2, 0), slider_2, lambda v: (-1 * math.sin(math.radians(v))))
    green.register_slider((1, 1), (2, 2), slider_2, lambda v: math.cos(math.radians(v)))

    slider_3 = widgets.Slider(pyplot.subplot(sliders[2, 0]), "Rotate: Z", 0, 360, 0, **style.darkgreen)
    green.register_slider((1, 2), (0, 0), slider_3, lambda v: math.cos(math.radians(v)))
    green.register_slider((1, 2), (0, 1), slider_3, lambda v: (-1 * math.sin(math.radians(v))))
    green.register_slider((1, 2), (1, 0), slider_3, lambda v: math.sin(math.radians(v)))
    green.register_slider((1, 2), (1, 1), slider_3, lambda v: math.cos(math.radians(v)))

    axes.relim()
    axes.autoscale_view()

    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":
    experiment()

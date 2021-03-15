#!/usr/bin/env python3

import logging
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
    # TODO: When is M^-1 =/= M^T?

    # TODO: Calculate distance from the camera at every pixel, and tint pixel to be lighter the farther it is from the camera.

    viewport_ratio = 4  # 3/4 rows for viewport, 1/4 rows for sliders
    num_sliders = 6

    figure: pyplot.Figure = pyplot.figure()
    grid: pyplot.GridSpec = gridspec.GridSpec(viewport_ratio, 1, figure=figure)
    axes: pyplot.Axes = pyplot.subplot(grid[:viewport_ratio - 1, :])

    sliders = gridspec.GridSpecFromSubplotSpec(num_sliders, 2, grid[-1, :], wspace=0.4)

    # Configure style
    axes.axis("equal")
    axes.grid(alpha=0.15, linestyle="--")
    axes.margins(2, 2)
    axes.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axes.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

    # Set up gray (baseline) patch
    origin = (0, 0)
    InteractiveSquare(axes, origin, style=style.gray)

    # Set up green (interactive) patch
    K = np.array([  # Intrinsic parameter matrix
        [1, 0, 0, 0],  # [αₓ γ  μ₀ 0]
        [0, 1, 0, 0],  # [0  αᵧ ν₀ 0]
        [0, 0, 1, 0]   # [0  0  1  0]
    ], dtype=float)

    Rx = np.identity(3, dtype=float)  # Extrinsic parameter matrix
    Ry = np.identity(3, dtype=float)  # [R,3x3 T,3x1]
    Rz = np.identity(3, dtype=float)  # [0,1x3     1]
    T = np.array([[0, 0, 0]], dtype=float)
    B = np.array([[0, 0, 0, 1]], dtype=float)

    green = InteractiveSquare(axes, (0, 0), 1, (0, 1), style=style.green,
                              convert_2d=utility.from_homogenous, label_vertices=True)

    green.register_transform(K, label="K")

    # Rx, Ry, Rz, T, B must be resolved before K can dot it.
    RT = Sequence()
    RT.register_node(MutableMatrix("Rz", Rz), None)
    RT.register_node(MutableMatrix("Ry", Ry), np.dot)
    RT.register_node(MutableMatrix("Rx", Rx), np.dot)
    RT.register_node(MutableMatrix("T", T), lambda a, b: np.concatenate((a, b.T), axis=1))
    RT.register_node(MutableMatrix("B", B), lambda a, b: np.concatenate((a, b), axis=0))
    green.register_transform(RT)

    # TODO: Do not register sliders to x, y, z angles. Register them to yaw (α), pitch (β), and roll (γ).
    #       Would need to register matrices/sliders with functions to calculate world rotations so that
    #       yaw, pitch, roll are valid--need to convert from intrinsic to extrinsic angles.

    # Yaw (α)
    slider_1 = widgets.Slider(pyplot.subplot(sliders[0, 0]), "Rotate: α", 0, 360, 0, **style.darkgreen)
    green.register_slider((1, 0), (0, 0), slider_1, lambda v: math.cos(math.radians(v)))
    green.register_slider((1, 0), (0, 1), slider_1, lambda v: (-1 * math.sin(math.radians(v))))
    green.register_slider((1, 0), (1, 0), slider_1, lambda v: math.sin(math.radians(v)))
    green.register_slider((1, 0), (1, 1), slider_1, lambda v: math.cos(math.radians(v)))

    # Pitch (β)
    slider_2 = widgets.Slider(pyplot.subplot(sliders[1, 0]), "Rotate: β", 0, 360, 0, **style.darkgreen)
    green.register_slider((1, 1), (0, 0), slider_2, lambda v: math.cos(math.radians(v)))
    green.register_slider((1, 1), (0, 2), slider_2, lambda v: math.sin(math.radians(v)))
    green.register_slider((1, 1), (2, 0), slider_2, lambda v: (-1 * math.sin(math.radians(v))))
    green.register_slider((1, 1), (2, 2), slider_2, lambda v: math.cos(math.radians(v)))

    # Roll (γ)
    slider_3 = widgets.Slider(pyplot.subplot(sliders[2, 0]), "Rotate: γ", 0, 360, 0, **style.darkgreen)
    green.register_slider((1, 2), (1, 1), slider_3, lambda v: math.cos(math.radians(v)))
    green.register_slider((1, 2), (1, 2), slider_3, lambda v: (-1 * math.sin(math.radians(v))))
    green.register_slider((1, 2), (2, 1), slider_3, lambda v: math.sin(math.radians(v)))
    green.register_slider((1, 2), (2, 2), slider_3, lambda v: math.cos(math.radians(v)))

    # Focal length
    slider_4 = widgets.Slider(pyplot.subplot(sliders[3, 0]), "Focal:", 0, 2, 0, **style.darkgreen)
    green.register_slider(0, (1, 0), slider_4)
    green.register_slider(0, (0, 1), slider_4)

    # Princible point x component
    slider_5 = widgets.Slider(pyplot.subplot(sliders[4, 0]), "PPx:", -5, 5, 0, **style.darkgreen)
    green.register_slider(0, (0, 2), slider_5)

    # Principle point y component
    slider_6 = widgets.Slider(pyplot.subplot(sliders[5, 0]), "PPy:", -5, 5, 0, **style.darkgreen)
    green.register_slider(0, (1, 2), slider_6)

    # World origin x component
    slider_7 = widgets.Slider(pyplot.subplot(sliders[0, 1]), "Tx", -5, 5, 0, **style.darkgreen)
    green.register_slider((1, 3), (0, 0), slider_7)

    # World origin y component
    slider_8 = widgets.Slider(pyplot.subplot(sliders[1, 1]), "Ty", -5, 5, 0, **style.darkgreen)
    green.register_slider((1, 3), (0, 1), slider_8)

    # World origin z component
    slider_9 = widgets.Slider(pyplot.subplot(sliders[2, 1]), "Tz", -5, 5, 1, **style.darkgreen)
    green.register_slider((1, 3), (0, 2), slider_9)

    logging.info(green.get_label())

    axes.relim()
    axes.autoscale_view()

    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    experiment()

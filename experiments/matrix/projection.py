#!/usr/bin/env python3

import math

import numpy as np
from matplotlib import gridspec, image, patches, pyplot, ticker, widgets
from numpy.linalg import norm

import src.style as style
import src.utility as utility
from src.interactivesquare import InteractiveSquare


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
    origin = (0.5, 0.5)
    gray = InteractiveSquare(origin, style=style.gray)
    axes.add_patch(gray.get_patch())

    # Set up green (interactive) patch
    K = np.array([  # Intrinsic paremeter matrix
        [1, 0, 0, 0],  # αₓ γ  μ₀ 0
        [0, 1, 0, 0],  # 0  αᵧ ν₀ 0
        [0, 0, 1, 0]   # 0  0  1  0
    ])
    RT = np.array([  # Extrinsic parameter matrix
        [1, 0, 0, 0],  # [R,3x3 T,3x1]
        [0, 1, 0, 0],  # [0,1x3 1    ]
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    Rx = np.identity(3)
    Ry = np.identity(3)
    Rz = np.identity(3)

    green = InteractiveSquare((0.5, 0.5), 1, style=style.green, callback_2d=utility.from_homogenous)
    axes.add_patch(green.get_patch())

    green.register_transform(0, K)
    green.register_transform(1, RT)

    slider_1 = widgets.Slider(pyplot.subplot(sliders[0, 0]), "Rotate: X", 0, 360, 0, **style.darkgreen)
    green.register_slider(1, (1, 1), slider_1, lambda v: math.cos(math.radians(v)))
    green.register_slider(1, (2, 2), slider_1, lambda v: math.cos(math.radians(v)))
    green.register_slider(1, (1, 2), slider_1, lambda v: (-1 * math.sin(math.radians(v))))
    green.register_slider(1, (2, 1), slider_1, lambda v: math.sin(math.radians(v)))

    slider_2 = widgets.Slider(pyplot.subplot(sliders[1, 0]), "Rotate: Y", 0, 360, 0, **style.darkgreen)
    green.register_slider(1, (0, 0), slider_2, lambda v: math.cos(math.radians(v)))
    green.register_slider(1, (2, 2), slider_2, lambda v: math.cos(math.radians(v)))
    green.register_slider(1, (2, 0), slider_2, lambda v: (-1 * math.sin(math.radians(v))))
    green.register_slider(1, (0, 2), slider_2, lambda v: math.sin(math.radians(v)))

    slider_3 = widgets.Slider(pyplot.subplot(sliders[2, 0]), "Rotate: Z", 0, 360, 0, **style.darkgreen)
    green.register_slider(1, (0, 0), slider_3, lambda v: math.cos(math.radians(v)))
    green.register_slider(1, (1, 1), slider_3, lambda v: math.cos(math.radians(v)))
    green.register_slider(1, (0, 1), slider_3, lambda v: (-1 * math.sin(math.radians(v))))
    green.register_slider(1, (1, 0), slider_3, lambda v: math.sin(math.radians(v)))

    axes.relim()
    axes.autoscale_view()

    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":
    experiment()

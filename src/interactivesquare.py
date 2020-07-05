import numpy as np
from matplotlib import patches, widgets

import src.utility as utility


"""
    blue_handle: patches.Patch = ax.add_patch(patches.Polygon(blue, **style.blue))

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
"""


class InteractiveSquare:
    def __init__(self, origin: tuple, scale: int, style: dict = None, transform_matrix: np.ndarray = None):
        self._matrix = transform_matrix

        self._indices = {}  # key: index tuple, value: index value
        self._square = utility.square(origin, scale)

        if style:
            self._patch = patches.Polygon(self._square, **style)
        else:
            self._patch = patches.Polygon(self._square)

    def _get_transform_matrix(self):
        """If _indices reference dimensions larger than the shape of _matrix, generate a new identity matrix and
        copy _matrix into it.

        TODO: But what about rectangular matrices?
            e.g.
                [1 0 0 0]
                [0 1 0 0]
                [0 0 1 0]

            I think I can just generate an NxN identity matrix, where N is the largest column dimension.
            Actually, maybe not--should the extra rows/columns be 0?
        """
        # Make a new matrix that will support all elements in self._matrix and from self._indices
        indices = self._indices.keys()
        max_y = max(index[1] for index in indices) + 1
        if self._matrix is not None:
            max_y = max(max_y, self._matrix.shape[1])
        matrix = np.identity(max_y, dtype=float)

        # Copy self._matrix into new matrix
        if self._matrix is not None:
            matrix[:self._matrix.shape[0], :self._matrix.shape[1]] = self._matrix

        # Copy values specified in self._indices into new matrix
        for index, value in self._indices.items():
            matrix[index] = value

        return matrix

    def _update_patch(self):
        points = utility.apply_transform(self._get_transform_matrix(), self._square)
        self._patch.set_xy(points)

    def _get_matrix_updater(self, index: tuple):
        def func(value: int):
            self._indices[index] = value
            self._update_patch()

        return func

    def get_patch(self):
        return self._patch

    def register_slider(self, index: tuple, slider: widgets.Slider):
        callback = self._get_matrix_updater(index)
        slider.on_changed(callback)
        callback(slider.valinit)

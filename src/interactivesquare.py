import numpy as np
from matplotlib import patches, widgets

import src.utility as utility


class InteractiveSquare:
    def __init__(self, origin: tuple = None, scale: int = 1, style: dict = None, transform_matrix: np.ndarray = None):
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

        TODO: But what about rectangular transform matrices?
            e.g.
                [1 0 0 0]
                [0 1 0 0]
                [0 0 1 0]

            I think I can just generate a square identity matrix with the largest matrix dimension.
        """
        # Make a new matrix that will support all elements in self._matrix and from self._indices
        max_rows = 0
        max_cols = 0

        indices = self._indices.keys()
        if self._indices:
            max_rows = max(index[0] for index in indices) + 1  # Shape is 1 greater than largest index
            max_cols = max(index[1] for index in indices) + 1

        if self._matrix is not None:
            max_rows = max(max_rows, self._matrix.shape[0])
            max_cols = max(max_cols, self._matrix.shape[1])

        dim = max(max_rows, max_cols)
        matrix = np.identity(dim, dtype=float)

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

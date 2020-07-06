import numpy as np
from matplotlib import patches, widgets

import src.utility as utility


class InteractiveSquare:
    def __init__(self, origin: tuple = None, scale: int = 1, style: dict = None, transform_matrix: np.ndarray = None):

        self._matrices = {}
        """
        {
            key (int): (np.ndarray, {
                (int, int): float
            })
        }
        """
        if transform_matrix is not None:
            self._matrices[0] = [transform_matrix, {}]

        self._square = utility.square(origin, scale)

        if style:
            self._patch = patches.Polygon(self._square, **style)
        else:
            self._patch = patches.Polygon(self._square)

    def _get_transform_matrix_component(self, order: int) -> np.ndarray:
        """If _indices reference dimensions larger than the shape of _matrix, generate a new identity matrix and
        copy _matrix into it.

        TODO: But what about rectangular transform matrices?
            e.g.
                [1 0 0 0]
                [0 1 0 0]
                [0 0 1 0]

            I think I can just generate a square identity matrix with the largest matrix dimension.
        """
        entry = self._matrices[order]
        matrix = entry[0]
        indices = entry[1]

        # Make a new matrix that will support all elements in self._matrix and from self._indices
        max_rows = 0
        max_cols = 0

        if indices:
            max_rows = max(index[0] for index in indices) + 1  # Shape is 1 greater than largest index
            max_cols = max(index[1] for index in indices) + 1

        if matrix is not None:
            max_rows = max(max_rows, matrix.shape[0])
            max_cols = max(max_cols, matrix.shape[1])

        dim = max(max_rows, max_cols)
        transform = np.identity(dim, dtype=float)

        # Copy self._matrix into new matrix
        if matrix is not None:
            transform[:matrix.shape[0], :matrix.shape[1]] = matrix

        # Copy values specified in self._indices into new matrix
        for index, value in indices.items():
            transform[index] = value

        return transform

    def _get_transform_matrix(self):
        # Iterate over all matrices backwards, because logically, applying transform A and then B:
        #   0: A
        #   1: B
        # must be applied like BAx where x is the point coordinate
        keys = sorted(self._matrices.keys(), reverse=True)

        if keys:
            transform = self._get_transform_matrix_component(keys[0])
            for order in keys[1:]:
                transform = np.dot(transform, self._get_transform_matrix_component(order))
        else:
            transform = np.array([])

        return transform

    def _update_patch(self):
        points = utility.apply_transform(self._get_transform_matrix(), self._square)
        self._patch.set_xy(points)

    def _get_matrix_updater(self, order: int, index: tuple):
        if order not in self._matrices:
            self._matrices[order] = [None, {}]

        def func(value: int):
            indices = self._matrices[order][1]
            indices[index] = value
            self._update_patch()

        return func

    def get_patch(self):
        return self._patch

    def register_transform(self, order: int, transform_matrix: np.ndarray):
        if order not in self._matrices:
            self._matrices[order] = [None, {}]

        self._matrices[order][0] = transform_matrix

    def register_slider(self, order: int, index: tuple, slider: widgets.Slider):
        callback = self._get_matrix_updater(order, index)
        slider.on_changed(callback)
        callback(slider.valinit)

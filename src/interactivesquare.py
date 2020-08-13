import typing

import numpy as np
from matplotlib import patches, widgets

import src.utility as utility


class InteractiveSquare:
    """Square interactable via transform matrices, and sliders which alter matrix specific-index values.
    """

    def __init__(self, origin: tuple = None, scale: float = 1, add_coords: typing.Iterable = None, style: dict = None,
                 transform_matrix: np.ndarray = None, callback_2d: typing.Callable = None):
        """Constructor

        Parameters
        ----------
        origin : tuple, optional
            2D center of square, by default (0, 0)

        scale : float, optional
            Scale of the square, by default 1

        add_coords : typing.Iterable, optional
            Include these additional coordinates in each point, by default None

        style : dict, optional
            Patch style, by default None

        transform_matrix : np.ndarray, optional
            Transformation matrix to apply to all points, by default None

        callback_2d: typing.Callable, optional
            Conversion function from the point's space to 2d space

        """
        self._matrices = {}
        # {
        #     key (int): (np.ndarray, {
        #         (int, int): float
        #     })
        # }

        if transform_matrix is not None:
            self._matrices[0] = [transform_matrix, {}]

        if callback_2d is None:
            self._callback_2d = self._first_two_coordinates
        else:
            self._callback_2d = callback_2d

        self._square = utility.square(origin, scale, add_coords=add_coords)

        if style:
            self._patch = patches.Polygon(self._square[:, :2], **style)
        else:
            self._patch = patches.Polygon(self._square[:, :2])

        self._update_index = 0
        self._update_patch()

    @staticmethod
    def _first_two_coordinates(point: np.ndarray) -> np.ndarray:
        return point[:, :2]

    def _get_transform_matrix_component(self, order: int) -> np.ndarray:
        # If _indices reference dimensions larger than the shape of _matrix, generate a new identity matrix and
        # copy _matrix into it.

        # TODO: But what about rectangular transform matrices?
        #     e.g.
        #         [1 0 0 0]
        #         [0 1 0 0]
        #         [0 0 1 0]

        #     I think I can just generate a square identity matrix with the largest matrix dimension.
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

    def _get_transform_matrix(self) -> np.ndarray:
        # Iterate over all matrices backwards, because logically, applying transform A and then B must be applied
        # like BAx where x is a point coordinate
        keys = sorted(self._matrices.keys(), reverse=True)

        if keys:
            transform = self._get_transform_matrix_component(keys[0])
            for order in keys[1:]:
                transform = np.dot(transform, self._get_transform_matrix_component(order))
        else:
            transform = np.array([])

        return transform

    def _update_patch(self):
        try:
            transform = self._get_transform_matrix()
            points = utility.apply_transform(transform, self._square)

            if points.shape[1] > 2:
                points = self._callback_2d(points)

            self._patch.set_xy(points)

        except IndexError:
            # Instance does not have data to update the patch with.
            pass

    def _update(self, _):
        return self._update_patch()

    def _get_matrix_updater(self, order: int, index: tuple, mutator: typing.Callable = None) -> typing.Callable:
        if order not in self._matrices:
            self._matrices[order] = [None, {}]

        def func(value: int):
            if mutator is not None:
                value = mutator(value)

            indices = self._matrices[order][1]
            indices[index] = value

        return func

    def get_patch(self) -> patches.Polygon:
        return self._patch

    def register_transform(self, order: int, transform_matrix: np.ndarray):
        if order not in self._matrices:
            self._matrices[order] = [None, {}]

        self._matrices[order][0] = transform_matrix
        self._update_patch()

    def register_slider(self, order: int, index: tuple, slider: widgets.Slider, mutator: typing.Callable = None):
        callback = self._get_matrix_updater(order, index, mutator)
        slider.on_changed(callback)

        # self._update must be called last! Disconnect and reconnect it if it was already registered.
        # Note: This will cause fragmentation of indices in the slider.
        if self._update in slider.observers.values():
            slider.disconnect(self._update_index)

        self._update_index = slider.on_changed(self._update)

        callback(slider.valinit)

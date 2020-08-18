import typing

import numpy as np
from matplotlib import patches, widgets

import src.utility as utility


class InteractiveSquare:
    """Square interactable via transform matrices, and sliders which alter matrix-index values.

    Square which can have multiple transformation matrices registered to it. Multiple registered matrices
    are coalesced into one matrix before being applied to the square. By default, matrices are coalesced
    using the dot product, but transforms can be registered alongside a coalescing function to override this.

    Matrices are registered in order starting from 0, but are coalesced in reverse order.
    This is because applying matrix A and then B is applied like (BA)x where x is the point vector.

    Sliders can be registered to control the value of any index of any component matrix.

    Example
    -------

    shear_x = np.array([
        [1, 0.5],
        [0,   1]
    ])
    shear_y = np.array([
        [1,   0],
        [0.5, 1]
    ])

    square = InteractiveSquare()
    square.register_transform(shear_x)

    #       (shear_x)
    #       [1, 0.5]   ->  np.dot  ->  [point_x]
    #       [0,   1]                   [point_y]

    square.register_transform(shear_y)

    #       (shear_y)                  (shear_x)
    #       [1,   0]   ->  np.dot  ->  [1, 0.5]   ->  np.dot  ->  [point_x]
    #       [0.5, 1]                   [0,   1]                   [point_y]

    square = InteractiveSquare()
    square.register_transform(shear_x)

    #       (shear_x)
    #       [1, 0.5]   ->  np.dot  ->  [point_x]
    #       [0,   1]                   [point_y]

    square.register_transform(shear_y, my_function)

    #       (shear_y)                       (shear_x)
    #       [1,   0]   ->  my_function  ->  [1, 0.5]  ->  np.dot  ->  [point_x]
    #       [0.5, 1]                        [0,   1]                  [point_y]

    my_slider = widgets.Slider(axes, "Shear X", 0, 1, 0.5)
    square.register_slider(0, (0, 1), my_slider)

    #       (shear_y)                       (shear_x)
    #       [1,   0]   ->  my_function  ->  [1, my_slider]  ->  np.dot  ->  [point_x]
    #       [0.5, 1]                        [0,         1]                  [point_y]

    """

    def __init__(self, origin: tuple = None, scale: float = 1, add_coords: typing.Iterable = None, style: dict = None,
                 convert_2d: typing.Callable = None):
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

        convert_2d: typing.Callable, optional
            Conversion function from the point's space to 2d space

        """
        self._matrices = {}
        # {
        #   key (int): [
        #       np.ndarray,             # Transform matrix
        #       { (int, int): float },  # Index into matrix: Value for index
        #       typing.Callable         # Coalescing function
        #   ]
        # }

        if convert_2d is None:
            self._convert_2d = self._first_two_coordinates
        else:
            self._convert_2d = convert_2d

        self._square = utility.square(origin, scale, add_coords=add_coords)

        if style:
            self._patch = patches.Polygon(self._square[:, :2], **style)
        else:
            self._patch = patches.Polygon(self._square[:, :2])

        self._num_matrices = 0
        self._update_index = 0
        self._update_patch()

    @staticmethod
    def _first_two_coordinates(point: np.ndarray) -> np.ndarray:
        return point[:, :2]

    def _get_transform_matrix_component(self, order: int) -> np.ndarray:
        """Returns transform matrix with index `order`

        Parameters
        ----------
        order : int
            Index of transform matrix

        Returns
        -------
        np.ndarray
            Transform matrix at index `order`
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

        # If _indices reference dimensions larger than the shape of _matrix, generate a new identity matrix and
        # copy _matrix into it.
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
        """Coalesces each component matrix to produce a single transform matrix

        Returns
        -------
        np.ndarray
            Single transformation matrix produced from all component matrices
        """
        # Iterate over all matrices backwards, because logically, applying transform A and then B must be applied
        # like BAx where x is a point coordinate
        components = sorted(self._matrices.items(), reverse=True)

        if components:
            transform = self._get_transform_matrix_component(components[0][0])
            coalesce = components[0][1][2]
        else:
            transform = np.array([])

        for component in components[1:]:
            key = component[0]
            data = component[1]

            transform = coalesce(transform, self._get_transform_matrix_component(key))

            coalesce = data[2]

        return transform

    def _update_patch(self):
        try:
            transform = self._get_transform_matrix()
            points = utility.apply_transform(transform, self._square)

            if points.shape[1] > 2:
                points = self._convert_2d(points)

            self._patch.set_xy(points)

        except IndexError:
            # Instance does not have data to update the patch with.
            pass

    def _update(self, _):
        return self._update_patch()

    def _get_matrix_updater(self, order: int, index: tuple, mutator: typing.Callable = None) -> typing.Callable:
        if order not in self._matrices:
            if order == 0:
                coalescer = None
            else:
                coalescer = np.dot

            self._matrices[order] = [None, {}, coalescer]

        def func(value: int):
            if mutator is not None:
                value = mutator(value)

            indices = self._matrices[order][1]
            indices[index] = value

        return func

    def get_patch(self) -> patches.Polygon:
        return self._patch

    def register_transform(self, transform_matrix: np.ndarray, coalescer: typing.Callable = None):
        order = self._num_matrices
        if order == 0:
            if coalescer is not None:
                raise ValueError("First transform has nothing to coalesce with!")
        else:
            if coalescer is None:
                coalescer = np.dot

        self._num_matrices += 1

        self._matrices[order] = [transform_matrix, {}, coalescer]
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

import typing

import numpy as np
from matplotlib import axes, patches, widgets

import src.utility as utility
from src.mutablematrix import MutableMatrix
from src.sequence import Sequence


class InteractiveSquare:
    """Square interactable via transform matrices, and sliders which alter matrix-index values.

    Square which can have multiple transformation matrices registered to it. Multiple registered matrices
    are coalesced into one matrix before being applied to the square. By default, matrices are coalesced
    using the dot product, but transforms can be registered alongside a coalescing function to override this.

    Matrices are registered in order starting from 0, but are coalesced in reverse order.
    This is because applying matrix A and then B is applied like (BA)x where x is the point vector.

    Sliders can be registered to control the value of any index of any component matrix.

    Parameters
    ----------
    axes : axes.Axes
        Axes object on which the square resides.

    origin : tuple, optional
        2D center of square, by default (0, 0).

    scale : float, optional
        Scale of the square, by default 1.

    add_coords : typing.Iterable, optional
        Include these additional coordinates in each point, by default None.

    style : dict, optional
        Patch style, by default None.

    convert_2d: typing.Callable[[np.ndarray], np.ndarray], optional
        Conversion function from the point's space to 2d space.

    label_vertices: bool, optional
        Label vertices of the square, by default False.

    Example
    -------
    ```python
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

    ```
    """

    def __init__(self, axes, origin=None, scale=1, add_coords=None, style=None, convert_2d=None, label_vertices=False):
        """Construct an instance."""
        self._sequence = Sequence()

        if convert_2d is None:
            self._convert_2d = self._first_two_coordinates
        else:
            self._convert_2d = convert_2d

        self._square = utility.square(origin, scale, add_coords=add_coords)

        if style:
            self._patch = patches.Polygon(self._square[:, :2], **style)
        else:
            self._patch = patches.Polygon(self._square[:, :2])

        self._labels = []
        if label_vertices:
            for (x, y), label in zip(self._patch.get_xy(), ["BL", "TL", "TR", "BR"]):
                text = axes.text(x, y, label)
                text.set_clip_on(True)
                self._labels.append(text)

        self._update_index = 0
        self._update_patch()

    @staticmethod
    def _first_two_coordinates(point):
        """Converts an N-dimensional point vector into a 2-dimensional point vector by truncating coordinates past the second

        Parameters
        ----------
        point : np.ndarray
            Point to convert.

        Returns
        -------
        np.ndarray
            2D point vector.
        """
        return point[:, :2]

    def _update_patch(self):
        """Update the square patch given the current transform matrix."""
        try:
            transform = self._sequence.get_matrix()
            points = utility.apply_transform(transform, self._square)

            if points.shape[1] > 2:
                points = self._convert_2d(points)

            self._patch.set_xy(points)
            if self._labels:
                for label, (x, y) in zip(self._labels, points):
                    label.set_x(x)
                    label.set_y(y)

        except ValueError:
            # Instance does not have data to update the patch with.
            pass

    def _update(self, _):
        """Callback function for slider.on_changed() that discards the given parameter."""
        self._update_patch()

    def get_patch(self):
        """Returns a patch for the square to register into an Axes object.

        Returns
        -------
        patches.Polygon
            Patch for the square.
        """
        return self._patch

    def register_transform(self, component, coalescer=None):
        """Register the transformation matrix as a component matrix.

        Parameters
        ----------
        component : np.ndarray
            Component to register.

        coalescer : typing.Callable[[np.ndarray, np.ndarray], np.ndarray], optional
            Function that coalesces this matrix with the previously registered matrix (if applicable),
            by default np.dot().

        Raises
        ------
        ValueError
            Raised when a coalescer is provided alongside the first transform matrix, as there are no prior
            matrices to coalesce with.

        """
        if coalescer is None:
            coalescer = np.dot

        if not isinstance(component, Sequence):
            component = MutableMatrix(component)

        self._sequence.register_node(component, coalescer)
        self._update_patch()

    def register_slider(self, index_of_component, index_within_component, slider, modifier=None):
        """Register a slider to control the value of matrix `order` at `index`

        Parameters
        ----------
        index_of_component : typing.Union[int, typing.Tuple[int]]
            Index of the matrix to select. Can be a series of indices to traverse into nested sequences.

        index_within_component : typing.Tuple[int]
            Index (R, C) of the value within the selected matrix.

        slider : widgets.Slider
            Slider to register.

        modifier : typing.Callable[[int], int], optional
            Callable function to mutate the slider value, by default None.
            e.g. to convert the slider value from degrees to radians.
        """
        if isinstance(index_of_component, int):
            index_of_component = (index_of_component,)

        node = self._sequence.get_node(index_of_component[0])
        for index in index_of_component[1:]:
            node = node.get_component().get_node(index)

        component = node.get_component()

        callback = component.get_mutator(index_within_component, modifier)
        slider.on_changed(callback)

        # self._update must be called last! Disconnect and reconnect it if it was already registered.
        # Note: This will cause fragmentation of indices in the slider.
        if self._update in slider.observers.values():
            slider.disconnect(self._update_index)

        self._update_index = slider.on_changed(self._update)

        callback(slider.valinit)

import typing

import numpy as np
from matplotlib import patches, widgets


def to_homogenous(array: np.ndarray) -> np.ndarray:
    """Convert each point in `array` to the homogenous coordinate space

    Parameters
    ----------
    array : np.ndarray
        Array of points to convert

    Returns
    -------
    np.ndarray
        Array of points converted to the homogenous coordinate space

    """
    return np.hstack((array, np.ones((array.shape[0], 1), dtype=array.dtype)))


def from_homogenous(array: np.ndarray) -> np.ndarray:
    """Convert each point in `array` to the cartesian coordinate space

    Parameters
    ----------
    array : np.ndarray
        Array of points to convert

    Returns
    -------
    np.ndarray
        Array of points converted to the cartesian coordinate space

    """
    rows = []
    for row in array:
        rows.append(row[:-1] / row[-1])

    return np.array(rows)


def square(origin: tuple = None, scale: float = 1, add_coords: typing.Iterable = None) -> np.ndarray:
    """Returns a square with scale `scale` centered around (`center_x`, `center_y`)

    Parameters
    ----------
    origin : tuple, optional
        2D center of square coordinate. Defaults to (0, 0), by default None

    scale : float, optional
        Scale of the square, by default 1

    add_coords : typing.Iterable, optional
        Include these additional coordinates in each point, by default None

    Returns
    -------
    np.ndarray
        Vector of points representing a square

    """
    if origin is None:
        # TODO: Should origin consider dimensions added by `add_coords`?
        origin = (0, 0)

    center_x = origin[0]
    center_y = origin[1]
    offset = scale / 2

    array: np.ndarray = np.array([
        [center_x - offset, center_y - offset],  # Bottom left
        [center_x - offset, center_y + offset],  # Top left
        [center_x + offset, center_y + offset],  # Top right
        [center_x + offset, center_y - offset]   # Bottom right
    ], dtype=float)

    if add_coords:
        cols: np.ndarray = np.array(add_coords, dtype=array.dtype)
        cols = np.tile(cols, (array.shape[0], 1))
        array = np.hstack((array, cols))

    return array


def transform(matrix: np.ndarray, row_vector: bool = False) -> typing.Callable[[np.ndarray], None]:
    """Returns a function which transforms points using the provided transformation `matrix` via matrix (dot) point.

    Parameters
    ----------
    matrix : np.ndarray
        Transformation matrix.

    row_vector : bool, optional
        `False` to treat points as column vectors, `True` to treat points as row vectors, by default False.

    Returns
    -------
    typing.Callable[[np.ndarray], None]
        Function which takes points and transforms them in-place.

    """
    if row_vector:
        # matrix, vector -> operate on column vectors
        # vector, matrix -> operate on row vectors
        # Tx = (xT)' = x'T'
        matrix = matrix.transpose()

    def func(point: np.ndarray):
        point_rows = point.shape[0]

        # if point has fewer rows than matrix columns, pad point column with rows containing 1
        delta = matrix.shape[1] - point_rows
        if delta > 0:
            coordinate = np.pad(point, (0, delta), constant_values=1)
        else:
            coordinate = point

        coordinate = np.dot(matrix, coordinate)[:point_rows]
        return coordinate

    return func


def apply_transform(transform_matrix: np.ndarray, points: np.ndarray, row_vector: bool = False) -> np.ndarray:
    """Applies `transform_matrix` to all points in array `points`

    Parameters
    ----------
    transform_matrix : np.ndarray
        Transformation matrix to apply to all points

    points : np.ndarray
        Array of points to transform

    row_vector : bool, optional
        `False` to treat points as column vectors, `True` to treat points as row vectors, by default False

    Returns
    -------
    np.ndarray
        Vector of transformed points

    """
    return np.apply_along_axis(transform(transform_matrix, row_vector=row_vector), 1, points)

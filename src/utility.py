import typing

import numpy as np


def to_homogenous(array: np.ndarray) -> np.ndarray:
    """Convert the input set of points to the homogenous coordinate space.

    Args:
        array (np.ndarray): Array of points to convert.

    Returns:
        np.ndarray: Array of points converted to the homogenous coordinate space.
    """
    return np.hstack((array, np.ones((array.shape[0], 1), dtype=array.dtype)))


def from_homogenous(array: np.ndarray) -> np.ndarray:
    """Convert the input set of points to the cartesian coordinate space.

    Args:
        array (np.ndarray): Array of points to convert.

    Returns:
        np.ndarray: Array of points converted to the cartesian coordinate space.
    """
    rows = []
    for row in array:
        rows.append(row[:-1] / row[-1])

    return np.array(rows)


def square(center_x: float = 0, center_y: float = 0, scale: float = 1, homogenous: bool = False) -> np.ndarray:
    """Returns a square with scale `scale` centered around (`center_x`, `center_y`).

    Args:
        center_x (float, optional): X axis centerpoint. Defaults to 0.
        center_y (float, optional): Y axis centerpoint. Defaults to 0.
        scale (float, optional): Scale of the square. Defaults to 1.
        homogenous (bool, optional): Include an extra coordinate for each point for projective geometry.
            Defaults to False.

    Returns:
        np.ndarray: Array of points representing a square.
    """
    offset = scale / 2

    array: np.ndarray = np.array([
        [center_x - offset, center_y - offset],  # Bottom left
        [center_x - offset, center_y + offset],  # Top left
        [center_x + offset, center_y + offset],  # Top right
        [center_x + offset, center_y - offset]   # Bottom right
    ], dtype=float)

    if homogenous:
        array = to_homogenous(array)

    return array


def transform(matrix: np.array, row_vector: bool = False) -> typing.Callable[[np.ndarray], None]:
    """Returns a function which transforms points using the provided transformation `matrix` via matrix (dot) point.

    Args:
        matrix (np.array): Transformation matrix
        row_vector (bool, optional): `False` to treat points as column vectors, `True` to treat points as row vectors.
            Defaults to False.

    Returns:
        typing.Callable[[np.ndarray], None]: Function which takes points and transforms them in-place.
    """
    if row_vector:
        # matrix, vector -> operate on column vectors
        # vector, matrix -> operate on row vectors
        # Tx = (xT)' = x'T'
        matrix = matrix.transpose()

    def func(point: np.ndarray):
        # if point has fewer columns that matrix rows, pad point columns with 1
        delta = matrix.shape[0] - point.shape[0]
        if delta > 0:
            coordinate = np.pad(point, (0, delta), constant_values=1)
        else:
            coordinate = point

        coordinate = np.dot(matrix, coordinate)

        point[:] = coordinate[:point.size]

    return func

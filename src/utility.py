import numpy as np


def square(center_x: float = 0, center_y: float = 0, scale: float = 1, homogenous: bool = False):
    offset = scale / 2

    array: np.ndarray = np.array([
        [center_x - offset, center_y - offset],  # Bottom left
        [center_x - offset, center_y + offset],  # Top left
        [center_x + offset, center_y + offset],  # Top right
        [center_x + offset, center_y - offset]   # Bottom right
    ], dtype=float)

    if homogenous:
        array = np.hstack((array, np.ones((array.shape[0], 1), dtype=array.dtype)))

    return array


def transform(matrix: np.array, row_vector: bool = False):
    if row_vector:
        # matrix, vector -> operate on column vectors
        # vector, matrix -> operate on row vectors
        # Tx = (xT)' = x'T'
        matrix = matrix.transpose()

    def func(point: np.array):
        # if point has fewer columns that matrix rows, pad point columns with 1
        delta = matrix.shape[0] - point.shape[0]
        if delta > 0:
            coordinate = np.pad(point, (0, delta), constant_values=1)
        else:
            coordinate = point

        coordinate = np.dot(matrix, coordinate)

        point[:] = coordinate[:point.size]

    return func

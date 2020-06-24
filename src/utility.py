import numpy as np


def square(center_x: float = 0, center_y: float = 0, scale: float = 1):
    offset = scale / 2

    return np.array([
        [center_x - offset, center_y - offset],  # Bottom left
        [center_x - offset, center_y + offset],  # Top left
        [center_x + offset, center_y + offset],  # Top right
        [center_x + offset, center_y - offset]   # Bottom right
    ], dtype=float)


def transform(matrix: np.array, row_vector=False):
    if row_vector:
        matrix = matrix.transpose()

    def func(point: np.array):
        # if point has fewer columns that matrix rows, pad point columns with 1
        delta = matrix.shape[0] - point.shape[0]
        if delta > 0:
            coordinate = np.pad(point, (0, delta), constant_values=1)
        else:
            coordinate = point

        # matrix, vector -> operate on column vectors
        # vector, matrix -> operate on row vectors
        coordinate = np.dot(matrix, coordinate)

        point[:] = coordinate[:point.size]

    return func

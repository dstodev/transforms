import numpy as np


def transform(matrix: np.array):
    def func(point: np.array):
        # if point has fewer columns that matrix rows, pad point columns with 1
        delta = matrix.shape[0] - point.shape[0]
        if delta > 0:
            coordinate = np.pad(point, (0, delta), constant_values=1)
        else:
            coordinate = point

        # vector, matrix -> operate on row vectors
        # matrix, vector -> operate on column vectors
        coordinate = np.dot(matrix, coordinate)
        point[:] = coordinate[:point.size]

    return func

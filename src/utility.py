import numpy as np


class Utility:

    @staticmethod
    def transform(matrix: np.array):
        def func(point: np.array):
            # if fewer point columns that matrix rows, pad point columns with 1
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


class Style:

    red = {
        "edgecolor": "#CC1010FF",
        "facecolor": "#FF808070"
    }

    green = {
        "edgecolor": "#10CC10FF",
        "facecolor": "#80FF8070"
    }

    blue = {
        "edgecolor": "#1010CCFF",
        "facecolor": "#8080FF70"
    }

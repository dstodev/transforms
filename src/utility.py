import numpy as np


class Utility:

    @staticmethod
    def transform(matrix: np.array):
        def func(point: np.array):
            # point, matrix -> operate on row vectors
            # matrix, point -> operate on col vectors
            point[:] = np.dot(point, matrix)

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

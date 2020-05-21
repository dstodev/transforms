import numpy as np


class Color:

    @staticmethod
    def shift(red: int, green: int, blue: int):
        scalars = np.array([red, green, blue])

        def func(rgb: np.array):
            rgb[:] = np.multiply(rgb, scalars)

        return func

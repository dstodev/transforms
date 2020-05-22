import numpy as np
from matplotlib import image, patches, pyplot

from src.affine import Affine
from src.utility import Style, Utility


def square():
    return np.array([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0]
    ], dtype=float)


if __name__ == "__main__":
    fig, ax = pyplot.subplots(figsize=(5, 5))
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)

    square_a = square()
    square_b = square()

    np.apply_along_axis(Affine.shear(0.5, 0), 1, square_a)
    np.apply_along_axis(Affine.shear(0, 0.5), 1, square_a)

    A = np.array([
        [1, 0.5],
        [0, 1]
    ])
    B = np.array([
        [1, 0],
        [0.5, 1]
    ])
    C = np.dot(A, B)

    np.apply_along_axis(Utility.transform(C), 1, square_b)

    ax.add_patch(patches.Polygon(square_a, **Style.blue))
    ax.add_patch(patches.Polygon(square_b, **Style.red))

    square_c = square()

    np.apply_along_axis(Affine.shear(1, 0), 1, square_c)

    ax.add_patch(patches.Polygon(square_c, **Style.green))

    pyplot.show()

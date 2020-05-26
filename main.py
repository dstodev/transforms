import numpy as np
from matplotlib import image, patches, pyplot

import src.affine as affine
import src.style as style
import src.utility as utility


def square(center_x: float = 0, center_y: float = 0, scale: float = 1):
    offset = scale / 2

    return np.array([
        [center_x - offset, center_y - offset],  # Bottom left
        [center_x - offset, center_y + offset],  # Top left
        [center_x + offset, center_y + offset],  # Top right
        [center_x + offset, center_y - offset]   # Bottom right
    ], dtype=float)


if __name__ == "__main__":
    fig, ax = pyplot.subplots(figsize=(5, 5))
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)

    origin_x = 0.5
    origin_y = 0.5

    A = np.array([
        [1, 0.5],
        [0, 1]
    ])
    B = np.array([
        [1, 0],
        [0.5, 1]
    ])
    C = np.dot(A, B)

    square_a = square(origin_x, origin_y)
    square_b = square(origin_x, origin_y)
    square_c = square(origin_x, origin_y)

    np.apply_along_axis(affine.shear(0.5, 0), 1, square_a)
    np.apply_along_axis(affine.shear(0, 0.5), 1, square_a)
    np.apply_along_axis(utility.transform(C), 1, square_b)
    np.apply_along_axis(affine.shear(1, 0), 1, square_c)

    ax.add_patch(patches.Polygon(square_a, **style.blue))
    ax.add_patch(patches.Polygon(square_b, **style.red))
    ax.add_patch(patches.Polygon(square_c, **style.green))

    pyplot.show()

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
    pyplot.grid(alpha=0.15, linestyle="--")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    origin_x = 0
    origin_y = 0

	# BATx == ?BAx

    A = np.array([
        [1, 0.5],
        [0, 1]
    ])
    B = np.array([
        [1, 0],
        [0.5, 1]
    ])
    C = np.dot(B, A)

    gray = square(origin_x, origin_y)
    red = square(origin_x, origin_y)
    green = square(origin_x, origin_y)
    blue = square(origin_x, origin_y)

	# (AB)' == B'A'
	# ABx --> xBA

	# xAB == (B'A'x')'

	# (xBA)' = B'A'x' = B'(A'x') ~= BAx in transposed coordinates

    np.apply_along_axis(affine.shear(1, 0), 1, red)

    np.apply_along_axis(utility.transform(C), 1, green)

    np.apply_along_axis(affine.shear(0.5, 0), 1, blue)
    np.apply_along_axis(affine.shear(0, 0.5), 1, blue)

    ax.add_patch(patches.Polygon(gray, **style.gray))
    ax.add_patch(patches.Polygon(red, **style.red))
    ax.add_patch(patches.Polygon(green, **style.green))
    ax.add_patch(patches.Polygon(blue, **style.blue))

    pyplot.show()

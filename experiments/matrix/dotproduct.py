import numpy as np
from numpy.linalg import norm


"""
Note:
    Matrix dimensions are typically denoted as M x N, with M rows and N columns.

    cos(theta) gives the x component of the coordinate pointed to by unit vector with angle theta
    arccos(adjacent/hypotenuse) gives theta of vector <x, y>

Matrix dot product:
    Two matrices MxN and PxQ:
        N must equal P
        Resultant matrix has shape MxQ
"""


def experiment():
    a = np.array([2, 1])
    b = np.array([2, 4])

    c = np.dot(a, b)

    theta = np.arccos(c / (norm(a) * norm(b)))


if __name__ == "__main__":
    experiment()

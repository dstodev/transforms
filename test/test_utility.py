from unittest import TestCase, mock

import numpy as np

from src.utility import (apply_transform, from_homogenous, square,
                         to_homogenous, transform)


class TestUtility(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_to_homogenous(self):
        expected = [[-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]]
        value_base = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        value_in = np.array(value_base, copy=True)

        actual = to_homogenous(value_in)

        self.assertEqual(value_in.tolist(), value_base)  # Assert input was not modified
        self.assertCountEqual(expected, actual.tolist())

    def test_from_homogenous(self):
        expected = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        value_base = [[-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]]
        value_in = np.array(value_base, copy=True)

        actual = from_homogenous(value_in)

        self.assertEqual(value_in.tolist(), value_base)
        self.assertEqual(expected, actual.tolist())

    def test_square(self):
        expected = [[-1, -1], [-1, 1], [1, -1], [1,  1]]
        actual = square(0, 0, 2)  # Square with width/height 2 centered around (0, 0)

        self.assertCountEqual(expected, actual.tolist())

    def test_square_homogenous(self):
        expected = [[-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]]
        actual = square(0, 0, 2, homogenous=True)

        self.assertCountEqual(expected, actual.tolist())

    def test_transform_matching_dimensions(self):
        expected = square(0, 0, 2)
        actual = square(0, 0, 2)

        T = np.array([
            [1, 0],
            [0, 1]
        ])

        np.apply_along_axis(transform(T), 1, actual)

        self.assertEqual(expected.tolist(), actual.tolist())

    def test_transform_smaller_coordinate(self):
        expected = square(0, 0, 2)
        actual = square(0, 0, 2)

        T = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        actual = np.apply_along_axis(transform(T), 1, actual)

        self.assertEqual(expected.tolist(), actual.tolist())

    def test_transform_affine_translation(self):
        expected = [[0, 1], [0, 3], [2, 1], [2, 3]]
        actual = square(0, 0, 2)

        T = np.array([
            [1, 0, 1],  # Translate 1 unit right
            [0, 1, 2],  # and 2 units up
            [0, 0, 1]
        ])
        actual = np.apply_along_axis(transform(T), 1, actual)

        self.assertCountEqual(expected, actual.tolist())

    def test_transform_row_vector(self):
        rect = [[0, 0], [0, 1], [2, 0], [2, 1]]
        expected = np.array(rect)
        actual = np.array(rect)

        Sx = np.array([[1, 0.25], [0, 1]])
        Sy = np.array([[1, 0], [0.5, 1]])
        T = np.dot(Sx, Sy)
        T_ = T.transpose()

        np.apply_along_axis(transform(T), 1, expected)
        np.apply_along_axis(transform(T_, row_vector=True), 1, actual)

        self.assertEqual(expected.tolist(), actual.tolist())

    def test_apply_transform_affine_translation(self):
        expected = [[0, 1], [0, 3], [2, 1], [2, 3]]
        actual = square(0, 0, 2)

        T = np.array([
            [1, 0, 1],  # Translate 1 unit right
            [0, 1, 2],  # and 2 units up
            [0, 0, 1]
        ])
        actual = apply_transform(T, actual)

        self.assertCountEqual(expected, actual.tolist())

    def test_apply_transform_row_vector(self):
        rect = [[0, 0], [0, 1], [2, 0], [2, 1]]
        expected = np.array(rect)
        actual = np.array(rect)

        Sx = np.array([[1, 0.25], [0, 1]])
        Sy = np.array([[1, 0], [0.5, 1]])
        T = np.dot(Sx, Sy)
        T_ = T.transpose()

        expected = apply_transform(T, expected)
        actual = apply_transform(T_, actual, row_vector=True)

        self.assertEqual(expected.tolist(), actual.tolist())

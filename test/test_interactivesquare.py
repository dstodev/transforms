from unittest import TestCase

import numpy as np

from src.interactivesquare import InteractiveSquare


class TestInteractiveSquare(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_instance(self):
        uut = InteractiveSquare()

        self.assertIsNotNone(uut)

    def test_instance_origin_scale(self):
        uut = InteractiveSquare(origin=(1, 1), scale=1)

        expected = [
            [0.5, 0.5],
            [0.5, 1.5],
            [1.5, 0.5],
            [1.5, 1.5]
        ]
        actual = uut._square

        self.assertCountEqual(expected, actual.tolist())

    def test_get_patch(self):
        uut = InteractiveSquare()

        expected = uut._patch
        actual = uut.get_patch()

        self.assertEqual(expected, actual)

    def test_get_transform_matrix_no_data(self):
        uut = InteractiveSquare()

        expected = []
        actual = uut._get_transform_matrix()

        self.assertEqual(expected, actual.tolist())

    def test_get_transform_matrix_with_only_matrix(self):
        expected = np.array([
            [1,   0.25],
            [0.5, 1]
        ])

        uut = InteractiveSquare(transform_matrix=expected)

        actual = uut._get_transform_matrix()

        self.assertEqual(expected.tolist(), actual.tolist())

    def test_get_transform_matrix_with_only_indices(self):
        uut = InteractiveSquare()
        uut._indices[(0, 1)] = 0.25
        uut._indices[(1, 0)] = 0.5

        expected = np.array([
            [1,   0.25],
            [0.5, 1]
        ])
        actual = uut._get_transform_matrix()

        self.assertEqual(expected.tolist(), actual.tolist())

    def test_get_transform_matrix_both_matrix_and_indices(self):
        matrix = np.array([
            [1, 0.25],
            [0, 1]
        ])

        uut = InteractiveSquare(transform_matrix=matrix)
        uut._indices[(1, 0)] = 0.5

        expected = np.array([
            [1,   0.25],
            [0.5, 1]
        ])
        actual = uut._get_transform_matrix()

        self.assertEqual(expected.tolist(), actual.tolist())

    def test_get_transform_matrix_horizontal_rectangular_matrix(self):
        matrix = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3]
        ])

        uut = InteractiveSquare(transform_matrix=matrix)

        expected = [
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ]
        actual = uut._get_transform_matrix()

        self.assertEqual(expected, actual.tolist())

    def test_get_transform_matrix_vertical_rectangular_matrix(self):
        matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 2, 3]
        ])

        uut = InteractiveSquare(transform_matrix=matrix)

        expected = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 2, 3, 1]
        ]
        actual = uut._get_transform_matrix()

        self.assertEqual(expected, actual.tolist())

    def test_get_matrix_updater(self):
        uut = InteractiveSquare()

        func = uut._get_matrix_updater((0, 1))
        func(0.5)

        expected = [
            [1, 0.5],
            [0, 1]
        ]
        actual = uut._get_transform_matrix()

        self.assertEqual(expected, actual.tolist())

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

    def test_get_transform_matrix_component_no_data(self):
        uut = InteractiveSquare()
        uut._matrices[0] = (None, {})

        expected = []
        actual = uut._get_transform_matrix_component(0)

        self.assertEqual(expected, actual.tolist())

    def test_get_transform_matrix_component_with_only_matrix(self):
        expected = np.array([
            [1,   0.25],
            [0.5, 1]
        ])

        uut = InteractiveSquare(transform_matrix=expected)

        actual = uut._get_transform_matrix_component(0)

        self.assertEqual(expected.tolist(), actual.tolist())

    def test_get_transform_matrix_component_with_only_indices(self):
        uut = InteractiveSquare()
        uut._matrices[0] = [None, {}]
        uut._matrices[0][1][(0, 1)] = 0.25
        uut._matrices[0][1][(1, 0)] = 0.5

        expected = np.array([
            [1,   0.25],
            [0.5, 1]
        ])
        actual = uut._get_transform_matrix_component(0)

        self.assertEqual(expected.tolist(), actual.tolist())

    def test_get_transform_matrix_component_both_matrix_and_indices(self):
        matrix = np.array([
            [1, 0.25],
            [0, 1]
        ])

        uut = InteractiveSquare(transform_matrix=matrix)
        uut._matrices[0][1][(1, 0)] = 0.5

        expected = np.array([
            [1,   0.25],
            [0.5, 1]
        ])
        actual = uut._get_transform_matrix_component(0)

        self.assertEqual(expected.tolist(), actual.tolist())

    def test_get_transform_matrix_component_horizontal_rectangular_matrix(self):
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
        actual = uut._get_transform_matrix_component(0)

        self.assertEqual(expected, actual.tolist())

    def test_get_transform_matrix_component_vertical_rectangular_matrix(self):
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
        actual = uut._get_transform_matrix_component(0)

        self.assertEqual(expected, actual.tolist())

    def test_get_matrix_updater(self):
        uut = InteractiveSquare()

        func = uut._get_matrix_updater(0, (0, 1))
        func(0.25)

        expected = [
            [1, 0.25],
            [0, 1]
        ]
        actual = uut._get_transform_matrix_component(0)

        self.assertEqual(expected, actual.tolist())

    def test_register_transform(self):
        uut = InteractiveSquare()

        matrix = np.array([
            [1,   0.25],
            [0.5, 1]
        ])

        uut.register_transform(1, matrix)

        expected = {1: [matrix, {}]}
        actual = uut._matrices

        self.assertEqual(expected, actual)

    def test_get_transform_matrix_no_data(self):
        uut = InteractiveSquare()

        expected = []
        actual = uut._get_transform_matrix()

        self.assertEqual(expected, actual.tolist())

    def test_get_transform_matrix_one_order_only_matrix(self):
        uut = InteractiveSquare()

        expected = np.array([
            [1,   0.25],
            [0.5, 1]
        ])

        uut.register_transform(1, expected)

        actual = uut._get_transform_matrix()

        self.assertEqual(expected.tolist(), actual.tolist())

    def test_get_transform_matrix_one_order_only_indices(self):
        uut = InteractiveSquare()
        uut._matrices[1] = (None, {(0, 1): 0.25})

        expected = [
            [1, 0.25],
            [0, 1]
        ]
        actual = uut._get_transform_matrix()

        self.assertEqual(expected, actual.tolist())

    def test_get_transform_matrix_one_order_both_matrix_and_indices(self):
        uut = InteractiveSquare()

        matrix = np.array([
            [1, 0.25],
            [0, 1]
        ])

        uut = InteractiveSquare(transform_matrix=matrix)
        uut._matrices[0][1][(1, 0)] = 0.5

        expected = np.array([
            [1,   0.25],
            [0.5, 1]
        ])
        actual = uut._get_transform_matrix()

        self.assertEqual(expected.tolist(), actual.tolist())

    def test_get_transform_matrix_two_order_only_matrix(self):
        matrix_1 = np.array([
            [1, 0.25],
            [0, 1]
        ])
        matrix_2 = np.array([
            [1,   0],
            [0.5, 1]
        ])

        uut = InteractiveSquare()
        uut._matrices[1] = (matrix_1, {})
        uut._matrices[2] = (matrix_2, {})

        expected = [
            [1.125, 0.25],
            [0.5,   1]
        ]
        actual = uut._get_transform_matrix()

        self.assertEqual(expected, actual.tolist())

    def test_get_transform_matrix_two_order_one_matrix_other_indices(self):
        matrix = np.array([
            [1, 0.25],
            [0, 1]
        ])

        uut = InteractiveSquare()
        uut._matrices[1] = (matrix, {})
        uut._matrices[2] = (None, {(1, 0): 0.5})

        expected = [
            [1.125, 0.25],
            [0.5,   1]
        ]
        actual = uut._get_transform_matrix()

        self.assertEqual(expected, actual.tolist())

    def test_get_transform_matrix_two_order_only_indices(self):
        uut = InteractiveSquare()
        uut._matrices[1] = (None, {(0, 1): 0.25})
        uut._matrices[2] = (None, {(1, 0): 0.5})

        expected = [
            [1.125, 0.25],
            [0.5,   1]
        ]
        actual = uut._get_transform_matrix()

        self.assertEqual(expected, actual.tolist())

    def test_get_transform_matrix_two_order_both_matrix_and_indices(self):
        matrix_1 = np.array([
            [1, 0.25],
            [0, 1]
        ])
        matrix_2 = np.array([
            [1,   0.75],
            [0, 1]
        ])

        uut = InteractiveSquare()
        uut._matrices[1] = (matrix_1, {(1, 0): 0.5})
        uut._matrices[2] = (matrix_2, {(1, 0): 1.25})

        expected = [
            [1.3125, 1],
            [1.75,   1.375]
        ]
        actual = uut._get_transform_matrix()

        self.assertEqual(expected, actual.tolist())

    def test_get_transform_matrix_three_order_sorted(self):
        matrix_1 = np.array([
            [0.25, 0.5],
            [0.75, 1]
        ])
        matrix_2 = np.array([
            [1.25, 1.5],
            [1.75, 2]
        ])
        matrix_3 = np.array([
            [2.25, 2.5],
            [2.75, 3]
        ])

        uut = InteractiveSquare()
        uut._matrices[1] = (matrix_1, {})
        uut._matrices[7] = (matrix_3, {})
        uut._matrices[3] = (matrix_2, {})

        expected = [
            [6.453125,  7.09375],
            [14.640625, 16.09375]
        ]
        actual = uut._get_transform_matrix()

        self.assertEqual(expected, actual.tolist())

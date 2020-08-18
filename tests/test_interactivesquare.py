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

        uut = InteractiveSquare()
        uut._matrices[0] = [expected, {}, None]

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

        uut = InteractiveSquare()
        uut._matrices[0] = [matrix, {}, None]
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

        uut = InteractiveSquare()
        uut._matrices[0] = [matrix, {}, None]

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

        uut = InteractiveSquare()
        uut._matrices[0] = [matrix, {}, None]

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

    def test_get_matrix_updater_mutator(self):
        uut = InteractiveSquare()

        func = uut._get_matrix_updater(0, (0, 1), lambda v: v + 1)
        func(1)

        expected = [
            [1, 2],
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

        uut.register_transform(matrix)

        expected = {0: [matrix, {}, None]}
        actual = uut._matrices

        self.assertEqual(expected, actual)

    def test_register_transform_with_coalescer(self):
        uut = InteractiveSquare()

        matrix_1 = np.array([
            [1,   0.25],
            [0.5, 1]
        ])
        matrix_2 = np.array([
            [1, 0],
            [0, 1]
        ])

        uut.register_transform(matrix_1)
        uut.register_transform(matrix_2)

        expected = {0: [matrix_1, {}, None], 1: [matrix_2, {}, np.dot]}
        actual = uut._matrices

        self.assertEqual(expected, actual)

    def test_register_transform_coalesce_with_nothing(self):
        uut = InteractiveSquare()

        matrix = np.array([
            [1,   0.25],
            [0.5, 1]
        ])

        with self.assertRaises(ValueError):
            uut.register_transform(matrix, coalescer=np.dot)

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

        uut.register_transform(expected)

        actual = uut._get_transform_matrix()

        self.assertEqual(expected.tolist(), actual.tolist())

    def test_get_transform_matrix_one_order_only_indices(self):
        uut = InteractiveSquare()
        uut._matrices[1] = (None, {(0, 1): 0.25}, None)

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

        uut = InteractiveSquare()
        uut._matrices[0] = [matrix, {}, None]
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
        uut._matrices[1] = (matrix_1, {}, np.dot)
        uut._matrices[2] = (matrix_2, {}, np.dot)

        expected = [
            [1,   0.25],
            [0.5, 1.125]
        ]
        actual = uut._get_transform_matrix()

        self.assertEqual(expected, actual.tolist())

    def test_get_transform_matrix_two_order_one_matrix_other_indices(self):
        matrix = np.array([
            [1, 0.25],
            [0, 1]
        ])

        uut = InteractiveSquare()
        uut._matrices[1] = (matrix, {}, np.dot)
        uut._matrices[2] = (None, {(1, 0): 0.5}, np.dot)

        expected = [
            [1,   0.25],
            [0.5, 1.125]
        ]
        actual = uut._get_transform_matrix()

        self.assertEqual(expected, actual.tolist())

    def test_get_transform_matrix_two_order_only_indices(self):
        uut = InteractiveSquare()
        uut._matrices[1] = (None, {(0, 1): 0.25}, np.dot)
        uut._matrices[2] = (None, {(1, 0): 0.5}, np.dot)

        expected = [
            [1,   0.25],
            [0.5, 1.125]
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
        uut._matrices[1] = (matrix_1, {(1, 0): 0.5}, np.dot)
        uut._matrices[2] = (matrix_2, {(1, 0): 1.25}, np.dot)

        expected = [
            [1.375, 1],
            [1.75,  1.3125]
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
        uut._matrices[1] = (matrix_1, {}, np.dot)
        uut._matrices[7] = (matrix_3, {}, np.dot)
        uut._matrices[3] = (matrix_2, {}, np.dot)

        expected = [
            [8.078125, 11.96875],
            [9.765625, 14.46875]
        ]
        actual = uut._get_transform_matrix()

        self.assertEqual(expected, actual.tolist())

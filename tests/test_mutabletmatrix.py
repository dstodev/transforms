from unittest import TestCase

from src.mutablematrix import MutableMatrix


class TestMutableMatrix(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_instance(self):
        matrix = [[1, 0], [0, 1]]

        uut = MutableMatrix("T", matrix)

        self.assertIsNotNone(uut)

    def test_get_matrix(self):
        matrix = [[1, 0], [0, 1]]

        uut = MutableMatrix("T", matrix)

        actual = uut.get_matrix().tolist()

        self.assertEqual(matrix, actual)

    def test_get_mutator(self):
        matrix = [[1, 0], [0, 1]]

        uut = MutableMatrix("T", matrix)

        mutator = uut.get_mutator((0, 0))
        mutator(2)

        expected = [[2, 0], [0, 1]]
        actual = uut.get_matrix().tolist()

        self.assertEqual(expected, actual)

    def test_get_mutator_with_modifier(self):
        matrix = [[1, 0], [0, 1]]

        uut = MutableMatrix("T", matrix)

        mutator = uut.get_mutator((1, 1), lambda v: v + 1)
        mutator(2)

        expected = [[1, 0], [0, 3]]
        actual = uut.get_matrix().tolist()

        self.assertEqual(expected, actual)

from unittest import TestCase

import numpy as np

from src.componentmatrix import ComponentMatrix
from src.sequence import Sequence


class MockComponent(ComponentMatrix):
    def __init__(self, matrix=None):
        if matrix is None:
            matrix = [[1, 0], [0, 1]]

        self._matrix = np.array(matrix)

    def get_matrix(self):
        return self._matrix


class TestSequence(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_instance(self):
        uut = Sequence()

        self.assertIsNotNone(uut)

    def test_get_node(self):
        uut = Sequence()

        component = MockComponent()
        coalescer = lambda: None

        uut.register_node(component, coalescer)

        self.assertEqual(1, len(uut._nodes))
        self.assertIs(component, uut.get_node(0).get_component())
        self.assertIs(coalescer, uut.get_node(0).get_coalescer())

    def test_register_node(self):
        uut = Sequence()

        component = MockComponent()
        coalescer = lambda: None

        uut.register_node(component, coalescer)

        self.assertEqual(1, len(uut._nodes))
        self.assertIs(component, uut.get_node(0).get_component())
        self.assertIs(coalescer, uut.get_node(0).get_coalescer())

    def test_get_matrix_no_elements(self):
        uut = Sequence()

        with self.assertRaises(ValueError):
            uut.get_matrix()

    def test_get_matrix_one_element(self):
        uut = Sequence()

        component = MockComponent()
        coalescer = lambda: None

        uut.register_node(component, coalescer)

        expected = component.get_matrix().tolist()
        actual = uut.get_matrix().tolist()

        self.assertEqual(expected, actual)

    def test_get_matrix_two_elements(self):
        uut = Sequence()

        uut.register_node(MockComponent(), None)
        uut.register_node(MockComponent(), np.add)

        expected = [[2, 0], [0, 2]]
        actual = uut.get_matrix().tolist()

        self.assertEqual(expected, actual)

    def test_get_matrix_nested_sequence(self):
        uut = Sequence()

        nested = Sequence()
        nested.register_node(MockComponent(), None)
        nested.register_node(MockComponent(), np.add)

        uut.register_node(MockComponent([[3, 0], [0, 3]]), None)
        uut.register_node(nested, np.multiply)

        expected = [[6, 0], [0, 6]]
        actual = uut.get_matrix().tolist()

        self.assertEqual(expected, actual)

from unittest import TestCase

from src.node import Node


class TestNode(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_instance(self):
        component = object()
        coalescer = lambda: None

        uut = Node(component, coalescer)

        self.assertIsNotNone(uut)

    def test_get_component(self):
        component = object()
        coalescer = lambda: None

        uut = Node(component, coalescer)

        self.assertIs(component, uut.get_component())

    def test_get_coalescer(self):
        component = object()
        coalescer = lambda: None

        uut = Node(component, coalescer)

        self.assertIs(coalescer, uut.get_coalescer())

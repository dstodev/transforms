import numpy as np

from src.componentmatrix import ComponentMatrix
from src.mutablematrix import MutableMatrix
from src.node import Node


class Sequence(ComponentMatrix):
    def __init__(self):
        self._nodes = []

    def get_matrix(self):
        matrix = self._nodes[0].get_component().get_matrix()

        for node in self._nodes[1:]:
            coalesce = node.get_coalescer()
            matrix = coalesce(matrix, node.get_component().get_matrix())

        return matrix

    def register_node(self, component, coalescer):
        """Register a node into the sequence.

        Parameters
        ----------
        component : ComponentMatrix
            Component to register.

        coalescer : typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
            Function to merge components in the sequence.

        """
        node = Node(component, coalescer)
        self._nodes.append(node)

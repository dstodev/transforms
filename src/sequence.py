import logging

import numpy as np

from src.componentmatrix import ComponentMatrix
from src.mutablematrix import MutableMatrix
from src.node import Node


class Sequence(ComponentMatrix):
    """Matrix sequence class.

    Manages a sequence of component matrices.

    """
    def __init__(self):
        """Construct an instance."""
        self._nodes = []

    def get_matrix(self):
        """Returns managed matrix.

        Returns a single matrix formed from the coalescence of all nodes.

        Returns
        -------
        np.ndarray
            Coalesced matrix.

        """
        try:
            lhs = self._nodes[0].get_component().get_matrix()
        except IndexError:
            raise ValueError("Sequence has no nodes!")

        for node in self._nodes[1:]:
            coalesce = node.get_coalescer()
            rhs = node.get_component().get_matrix()
            lhs = coalesce(lhs, rhs)

        return lhs

    def get_label(self):
        """Get string representation of node relationship."""
        label = ""

        enclosed = (len(self._nodes) > 1)
        if enclosed:
            label += "("

        try:
            lhs = self._nodes[0].get_component().get_label()
            label += lhs
        except IndexError:
            raise ValueError("Sequence has no nodes!")

        for node in self._nodes[1:]:
            label += " → "
            coalescer = node.get_coalescer()
            if coalescer:
                if coalescer.__name__ == "<lambda>":
                    label += "lambda"
                else:
                    label += coalescer.__qualname__
                label += " → "

            rhs = node.get_component().get_label()
            label += rhs

        if enclosed:
            label += ")"

        return label

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

    def get_node(self, index):
        """Gets node at `index`.

        Parameters
        ----------
        index : int
            Index of node to get.

        Returns
        -------
        Node
            Node at index.

        """
        return self._nodes[index]

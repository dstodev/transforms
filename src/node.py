import typing

import numpy as np

from src.componentmatrix import ComponentMatrix


class Node:
    def __init__(self, component, coalescer):
        """Node in a sequence.

        Parameters
        ----------
        component : ComponentMatrix
            Matrix to use as a component of the sequence.

        coalescer : typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
            Function to merge this matrix with the previous matrix in the sequence.

        """
        self._component = component
        self._coalescer = coalescer

    def get_component(self):
        """Get component matrix."""
        return self._component

    def get_coalescer(self):
        """Get matrix coalescer."""
        return self._coalescer

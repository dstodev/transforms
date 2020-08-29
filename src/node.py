import typing

import numpy as np

Coalescer = typing.Callable[[np.ndarray, np.ndarray], np.ndarray]  # Coalesces two matrices into one matrix


class Node:
    def __init__(self):
        self._component = None
        self._coalescer = None

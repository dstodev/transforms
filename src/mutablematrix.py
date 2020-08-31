import typing

import numpy as np

from src.componentmatrix import ComponentMatrix


class MutableMatrix(ComponentMatrix):
    """Manages a matrix object which can be modified by "mutator" functions.

    Manages an 2D np.ndarray (matrix) which can have its values mutated on a
    per-index basis through functions returned by `get_mutator()`.

    Parameters
    ----------
    matrix : typing.Iterable, optional
        Matrix to manage, by default None

    """

    def __init__(self, matrix=None):
        """Construct an instance."""
        if matrix is None:
            matrix = []

        self._matrix = np.array(matrix, ndmin=2)

    def get_matrix(self) -> np.ndarray:
        """Get managed matrix."""
        return self._matrix

    def get_mutator(self, index, modifier=None):
        """Returns a function which sets the `index` of the managed matrix to the
        value it is given.

        Returns a 'mutator' function which sets the matrix value at `index` when
        called with a value. Can provide a `modifier` callable which will modify
        the input value before setting the value at `index`.

        Parameters
        ----------
        index : typing.Tuple[int, int]
            Index into the matrix.

        modifier : typing.Callable[[float], float], optional
            Callable to modify values before setting, by default None.

        Returns
        -------
        typing.Callable[[float], None]
            Function which will set values at `index` when called.

        Example
        -------
        ```python
        >>> mm = MutableMatrix([
        ...     [1, 0],
        ...     [0, 1]
        ... ])

        >>> mutator = mm.get_mutator((0, 1))
        >>> mutator(2)
        >>> mm.get_matrix().tolist()  # doctest: +NORMALIZE_WHITESPACE
        [[1, 2],
         [0, 1]]

        >>> mutator = mm.get_mutator((1, 0), modifier=lambda v: v + 1)
        >>> mutator(2)
        >>> mm.get_matrix().tolist()  # doctest: +NORMALIZE_WHITESPACE
        [[1, 2],
         [3, 1]]

        ```
        """
        if modifier is None:
            def mutate(value: float):
                self._matrix[index] = value
        else:
            def mutate(value: float):
                self._matrix[index] = modifier(value)

        return mutate

import typing

import numpy as np

from src.componentmatrix import ComponentMatrix


class MutableMatrix(ComponentMatrix):
    def __init__(self, matrix=None, override_mode: str = "by_ref"):
        if matrix is None:
            matrix = []

        self._matrix = np.array(matrix, ndmin=2)
        self._overrides = {}

        if override_mode not in {"by_val", "by_ref"}:
            raise ValueError("Override mode must be either 'by_val' or 'by_ref'!")

        self._override_mode = override_mode

    def get_matrix(self) -> np.ndarray:
        if self._override_mode == "by_val":
            matrix = self._matrix.copy()
        else:
            matrix = self._matrix

        for index, value in self._overrides.items():
            matrix[index] = value

        return matrix

    def get_mutator(self, index, modifier=None):
        """Returns a function which sets `index` to the value it is given.

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

        >>> mutator = mm.get_mutator((1, 0), modifier=lambda v: v+1)
        >>> mutator(2)
        >>> mm.get_matrix().tolist()  # doctest: +NORMALIZE_WHITESPACE
        [[1, 2],
         [3, 1]]

        ```
        """
        if modifier is None:
            def mutate(value: float):
                self._overrides[index] = value
        else:
            def mutate(value: float):
                self._overrides[index] = modifier(value)

        return mutate
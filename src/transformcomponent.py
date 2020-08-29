import numpy as np


class TransformComponent:
    def __init__(self, override_mode: str = "by_ref"):
        self._matrix = np.array([], ndmin=2)
        self._overrides = []

        if override_mode not in {"by_val", "by_ref"}:
            raise ValueError("Override mode must be either 'by_val' or 'by_ref'!")

        self._override_mode = override_mode

    def get_matrix(self) -> np.ndarray:
        if self._override_mode == "by_val":
            matrix = self._matrix.copy()
        else:
            matrix = self._matrix

        for override in self._overrides:
            pass

        return matrix

import abc

import numpy


class ComponentMatrix(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_matrix(self) -> numpy.ndarray:
        """Get managed matrix."""
        raise NotImplementedError

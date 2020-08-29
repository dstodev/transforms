import abc


class ComponentMatrix(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_matrix(self):
        raise NotImplementedError

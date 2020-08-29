import numpy as np

from src.componentmatrix import ComponentMatrix
from src.mutablematrix import MutableMatrix
from src.node import Coalescer, Node


class Sequence(ComponentMatrix):
    def __init__(self):
        self._nodes = []

    def get_matrix(self):
        pass

    def register_node(self, component: MutableMatrix, coalescer: Coalescer):
        pass

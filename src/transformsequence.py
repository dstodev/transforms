from src.transformcomponent import TransformComponent
from src.transformnode import TransformNode
from src.transformtypes import Coalescer


class TransformSequence:
    def __init__(self):
        self._nodes = []

    def get_matrix(self):
        pass

    def register_node(self, component: TransformComponent, coalescer: Coalescer):
        pass

class TransformNode:
    def __init__(self):
        self._component = None  # Can be a TransformComponent or a TransformSequence
        self._coalescer = None

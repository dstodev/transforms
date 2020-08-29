import typing

import numpy as np

Coalescer = typing.Callable[[np.ndarray, np.ndarray], np.ndarray]

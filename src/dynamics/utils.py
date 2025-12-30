import numpy as np
from functools import cache
from .models import ActionMode


@cache
def get_directions(action_mode: ActionMode) -> np.ndarray:
    if action_mode == ActionMode.SIMPLE:
        return np.array([
            [-1, 0],  # up
            [0, 1],   # right
            [1, 0],   # down
            [0, -1]   # left
        ])

    if action_mode == ActionMode.CARDINAL:
        return np.array([
            [-1,  0], [-1,  1],
            [ 0,  1], [ 1,  1],
            [ 1,  0], [ 1, -1],
            [ 0, -1], [-1, -1]
        ])

    raise ValueError("Invalid ActionMode")

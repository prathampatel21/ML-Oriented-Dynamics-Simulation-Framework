from dataclasses import dataclass
from enum import Enum
import numpy as np


class ActionMode(Enum):
    SIMPLE = 4      # Up, Right, Down, Left
    CARDINAL = 8    # Includes diagonals


@dataclass(frozen=True)
class State:
    r: int
    c: int

    def numpy(self) -> np.ndarray:
        return np.array([self.r, self.c]).reshape((1, 2))

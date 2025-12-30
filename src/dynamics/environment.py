import numpy as np
from .models import ActionMode


class Environment:
    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        action_mode: ActionMode,
        discrete_rewards: bool = False,
        **kwargs
    ):
        self.m, self.n = n_rows, n_cols
        self.action_mode = action_mode
        self.board = np.zeros((self.m, self.n)) - 1

        if discrete_rewards:
            required = ["obstacle_count", "goal_count", "obstacle_value", "goal_value"]
            for k in required:
                if k not in kwargs:
                    raise KeyError(f"Missing `{k}` for discrete rewards")

            obs_idx = np.random.choice(self.board.size, kwargs["obstacle_count"], replace=False)
            np.put(self.board, obs_idx, kwargs["obstacle_value"])

            goal_idx = np.random.choice(self.board.size, kwargs["goal_count"], replace=False)
            np.put(self.board, goal_idx, kwargs["goal_value"])
        else:
            self.board = np.random.uniform(-1, 1, (self.m, self.n))

        self.dynamics = self._build_dynamics()

    def _build_dynamics(self) -> np.ndarray:
        A = self.action_mode.value
        dynamics = np.random.rand(self.m, self.n, A, A)

        for a in range(A):
            dynamics[:, :, a, a] += 100

        k = 1 if self.action_mode == ActionMode.SIMPLE else 2

        dynamics[0, :, :, 0 * k] = 0
        dynamics[:, -1, :, 1 * k] = 0
        dynamics[-1, :, :, 2 * k] = 0
        dynamics[:, 0, :, 3 * k] = 0

        dynamics /= dynamics.sum(axis=-1, keepdims=True)
        return dynamics

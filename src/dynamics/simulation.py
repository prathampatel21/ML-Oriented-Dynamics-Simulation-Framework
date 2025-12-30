import numpy as np
from .models import State
from .utils import get_directions


class Simulator:
    def __init__(self, env):
        self.env = env
        self.directions = get_directions(env.action_mode)

    def get_next_states(self, state: State) -> np.ndarray:
        base = state.numpy()
        candidates = base + self.directions
        mask = (
            (candidates[:, 0] >= 0) & (candidates[:, 0] < self.env.m) &
            (candidates[:, 1] >= 0) & (candidates[:, 1] < self.env.n)
        )
        return candidates[mask]

    def get_rewards(self, state: State) -> np.ndarray:
        nxt = self.get_next_states(state)
        return self.env.board[nxt[:, 0], nxt[:, 1]]

    def get_transition_probs(self, state: State, action: int) -> np.ndarray:
        probs = self.env.dynamics[state.r, state.c, action]
        return probs[probs > 0]

    def possible_actions(self, state: State) -> list[int]:
        actions = []
        for i, d in enumerate(self.directions):
            r, c = state.r + d[0], state.c + d[1]
            if 0 <= r < self.env.m and 0 <= c < self.env.n:
                actions.append(i)
        return actions

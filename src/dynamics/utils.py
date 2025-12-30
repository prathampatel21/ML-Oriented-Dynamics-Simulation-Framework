import numpy as np
from functools import cache

from dataclasses import dataclass

from enum import Enum
class ActionMode(Enum):
    SIMPLE = 4  # 4 directions: Up, Right, Down, Left
    CARDINAL = 8  # 8 directions: Includes diagonals

@dataclass
class State:
    r: int
    c: int

    def numpy(self):
        return np.array([self.r, self.c]).reshape((1, 2))


class Environment:
    def __init__(self, n_rows: int, n_cols: int, action_mode: ActionMode, discrete_rewards: bool = False, **kwargs):
        self.board = np.zeros((n_rows, n_cols)) - 1
        self.action_mode = action_mode
        self.m, self.n = n_rows, n_cols

        if discrete_rewards:
            if "obstacle_count" not in kwargs:
                raise KeyError("Did not provide `obstacle_count` when using discrete_rewards")
            if "goal_count" not in kwargs:
                raise KeyError("Did not provide `goal_count` when using discrete_rewards")
            if "obstacle_value" not in kwargs:
                raise KeyError("Did not provide `obstacle_value` when using discrete_rewards")
            if "goal_value" not in kwargs:
                raise KeyError("Did not provide `goal_value` when using discrete_rewards")
            # add -1 spots
            indices_obstacles = np.random.choice(self.board.size, size=kwargs["obstacle_count"], replace=False)
            np.put(self.board, indices_obstacles, np.take(self.board, indices_obstacles) + kwargs["obstacle_value"] + 1)

            # add to board
            indices_plus_1 = np.random.choice(self.board.size, size=kwargs["goal_count"], replace=False)
            np.put(self.board, indices_plus_1, np.take(self.board, indices_plus_1) + kwargs["goal_value"] + 1)
            
        else:
            self.board = np.random.uniform(-1, 1, (self.m, self.n))

        dynamics = np.random.rand(self.m, self.n, self.action_mode.value, self.action_mode.value) # T(s, a, s')

        # ensure the direction you are trying to take is always the highest
        for a in range(self.action_mode.value):
            dynamics[:, :, a, a] += 100

        # handle the borders as invalid action
        # probability of 0's
        k = None
        if self.action_mode == ActionMode.SIMPLE:
            k = 1
        elif self.action_mode == ActionMode.CARDINAL:
            k = 2
            # top right - top can't and right can't
            dynamics[0, :, :, 1] = 0
            dynamics[:, -1, :, 1] = 0

            # bottom right
            dynamics[-1, :, :, 3] = 0
            dynamics[:, -1, :, 3] = 0

            # bottom left
            dynamics[-1, :, :, 5] = 0
            dynamics[:, 0, :, 5] = 0


            # top left
            dynamics[0, :, :, 7] = 0
            dynamics[:, 0, :, 7] = 0            
        else:
            raise ValueError("invalid action mode provided")
        dynamics[0, :, :, 0 * k] = 0 # top
        dynamics[:, -1, :, 1 * k] = 0 # right
        dynamics[-1, :, :, 2 * k] = 0 # bottom
        dynamics[:, 0, :, 3 * k] = 0 # left

        # Normalize the probability distributions so that each action's distribution sums to 1
        dynamics /= dynamics.sum(axis=-1, keepdims=True)
        self.dynamics = dynamics

    def get_next_state_rewards(self, state: State, action_val: int) -> np.array:
        """
        Rewards for the next possible states.

        args:
            state: State - position of the agent on the board. Coordinates represented by (r, c)
            action: int - action index the agent wants to take (reference `directions()` to check you are using the correct one)
        
        returns:
            np.array representing rewards for all possible VALID next states. Note, any invalid border actions
            (i.e. moving up when in the top row) will not be included in this returned array.
        """
        next_states = self.get_next_states(state, action_val)
        rows, cols = next_states[:, 0], next_states[:, 1]
        
        rewards = self.board[rows, cols]
        return rewards
    
    def get_next_states(self, state: State, action_val: int) -> np.array:
        """
        Coordinates of next possible states from the current one.

        args:
            state: State -  position of the agent on the board. Coordinates represented by (r, c)
            action_val: int - action index the agent wants to take (reference `directions()` to check you are using the correct one)
        
        returns:
            np.array containing subarrays of size 2 for coordinates of possible actions
                the agent can move. There is a filter to ensure that only VALID next states are included.
        """
        
        state_npy = state.numpy()
        resulting_positions = np.repeat(state_npy, self.action_mode.value, axis=0) + self.directions # [a, 2]
        valid_positions_mask = (resulting_positions[:, 0] >= 0) & (resulting_positions[:, 0] < self.m) \
                & (resulting_positions[:, 1] >= 0) & (resulting_positions[:, 1] < self.n)
    
        return resulting_positions[valid_positions_mask]


    def get_transition_dynamics(self, state, action_val) -> np.array:
        """
        Probability distribution of next states from a particular state.

        args:
            state: State - position of the agent on the board. Coordinates represented by (r, c)
            action_val: int - action index the agent wants to take (reference `directions()` to check you are using the correct one)
        
        returns:
            np.array containing probability of arriving at each of the VALID next states possible. Note, this contains a filter for any
                actions that are not to be taken. The size of the array will depends on the ACTION_MODE
        """
        dynamics = self.dynamics[state.r, state.c, action_val]
        dynamics = dynamics[dynamics > 0]
        return dynamics
    
    def get_possible_actions(self, state: State) -> list[int]:
        """
        Generator for all action values possible in the current state.

        args:
            state: State - position of the agent on the board. Coordinates represented by (r, c)
        returns:
            yield value representing the action (see `directions()` to understand returned actions)
        """
        actions = []
        for dir_i in range(len(self.directions)):
            result_position = state.numpy() + self.directions[dir_i]
            if result_position[0, 0] < 0 or result_position[0, 0] >= self.m or result_position[0, 1] < 0 or result_position[0, 1] >= self.n:
                continue
                
            actions.append(dir_i)
        return actions

    def get_all_states(self):
        """
        Generator for all possible states on the board

        returns:
            yield State(r, c)
        """
        for r in range(self.m):
            for c in range(self.n):
                yield State(r=r, c=c)
    
    @property
    @cache
    def directions(self) -> np.array:
        if self.action_mode == ActionMode.SIMPLE:
                                # up        right   down    left
            directions = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        elif self.action_mode == ActionMode.CARDINAL:
            directions = np.array([
                [-1,  0],  # North
                [-1,  1],  # Northeast
                [ 0,  1],  # East
                [ 1,  1],  # Southeast
                [ 1,  0],  # South
                [ 1, -1],  # Southwest
                [ 0, -1],  # West
                [-1, -1]   # Northwest
            ])
        else:
            raise ValueError("Illegal action space dimension provided.")
        
        return directions
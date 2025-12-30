# ML-Oriented Dynamics Simulation Framework

A modular Python framework for modeling and simulating **discrete-time Markov Decision Processes (MDPs)**.
The project focuses on clean abstractions for state, action, reward, and transition dynamics, enabling
reproducible experimentation for reinforcement learning and control-oriented machine learning research.

## Motivation

Many ML prototypes tightly couple environment logic, transition dynamics, and experimentation code,
making it difficult to reason about system behavior or extend environments for new algorithms.

This project explores a cleaner design that separates:
- **state and action representations**
- **environment construction**
- **stochastic transition dynamics**
- **simulation queries used by learning algorithms**
- **experimental analysis**

The result is a lightweight but extensible simulation framework suitable for RL algorithm development,
model-based learning, and dynamics analysis.

## ML Framing

The environment explicitly models the core components of an MDP:

- **State (`S`)**: grid-based state representation
- **Action (`A`)**: configurable discrete action spaces (cardinal or diagonal)
- **Transition (`T(s, a, s')`)**: stochastic dynamics with normalized probability distributions
- **Reward (`R(s')`)**: continuous or discretized reward surfaces

This structure makes the system immediately usable for:
- policy evaluation
- value iteration
- planning algorithms
- model-based RL experiments

## Architecture Overview

src/dynamics/
├── models.py # State and action abstractions
├── environment.py # Environment + transition dynamics generation
├── simulation.py # Transition queries used by learning algorithms
├── utils.py # Cached geometry and shared helpers
└── io.py # Data loading utilities


Key design decisions:
- Immutable state objects to prevent unintended side effects
- Cached action geometry for efficient transition queries
- Explicit separation between environment generation and simulation logic
- Interfaces designed to be consumed by ML algorithms, not notebooks

## Features

- Discrete-time grid-based MDPs with configurable action spaces
- Stochastic transition dynamics with guaranteed probability normalization
- Support for both cardinal and diagonal action modes
- Modular APIs for:
  - valid action enumeration
  - next-state generation
  - reward lookup
  - transition probability queries
- Experiment notebooks demonstrating environment behavior and dynamics

## Example Usage

```python
from dynamics.models import State, ActionMode
from dynamics.environment import Environment
from dynamics.simulation import Simulator

env = Environment(
    n_rows=10,
    n_cols=10,
    action_mode=ActionMode.CARDINAL
)

sim = Simulator(env)
state = State(5, 5)

actions = sim.possible_actions(state)
next_states = sim.get_next_states(state)
rewards = sim.get_rewards(state)

```

These interfaces are intentionally designed to plug directly into
policy iteration, value iteration, or sampling-based reinforcement
learning pipelines.

## Experiments

Notebooks in `notebooks/` explore:

- Transition stochasticity under different action spaces
- Reward surface structure and spatial distributions
- Boundary effects and invalid action handling
- Empirical validation of transition probability distributions

All experiments are intentionally decoupled from core logic to preserve
testability and reproducibility.

## Testing

Unit tests in `tests/` validate:

- Correctness of transition dynamics
- Action validity at environment boundaries
- Probability normalization invariants

Tests are deterministic and written using `pytest`, enabling safe
refactors and reliable extension for learning-based experiments.

## Future Extensions

- Policy evaluation and planning algorithms (e.g., value iteration)
- Model-based reinforcement learning with learned transition dynamics
- Vectorized rollouts for batch learning and scalability
- Continuous-time extensions and differentiable dynamics
- Integration with standard reinforcement learning benchmarks and APIs

## Notes

This project prioritizes clarity of abstractions and correctness of
dynamics over environment-specific heuristics. The architecture is
intentionally designed to scale toward more complex machine learning
and reinforcement learning systems.

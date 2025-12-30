# Dynamics Simulation Framework

A modular Python framework for modeling and simulating discrete-time dynamical systems.
This project emphasizes clean abstractions between environment construction, state
representation, and transition dynamics, enabling reproducible experiments and
extensible system analysis.

## Motivation

Many simulation and reinforcement-learning prototypes tightly couple environment
definitions with transition logic, making experimentation brittle and hard to extend.
This project explores a cleaner architecture that separates:
- **system definition**
- **transition dynamics**
- **simulation queries**
- **experimental analysis**

The result is a small but extensible framework suitable for research, algorithm
development, and systems experimentation.

## Architecture Overview

src/dynamics/
├── models.py # State and action abstractions
├── environment.py # Environment construction and dynamics generation
├── simulation.py # State transition queries and rollout logic
├── utils.py # Shared geometry and cached helpers
└── io.py # Data loading utilities


Key design choices:
- Immutable state representation
- Cached direction geometry for fast queries
- Explicit separation between environment generation and simulation logic
- Testable, deterministic interfaces

## Features

- Discrete-time grid-based environments with configurable action spaces
- Stochastic transition dynamics with normalized probability distributions
- Support for cardinal and diagonal action modes
- Modular simulation queries (next states, rewards, valid actions)
- Experiment notebooks demonstrating system behavior

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

next_states = sim.get_next_states(state)
rewards = sim.get_rewards(state)

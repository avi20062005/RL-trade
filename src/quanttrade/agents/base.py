"""Agent abstraction.

A minimal :class:`Agent` protocol decouples the backtester and inference layer
from any specific learning library. Baseline agents are dependency-free so the
evaluation stack is fully testable without the deep-learning extras.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from quanttrade.env.portfolio import Action


@runtime_checkable
class Agent(Protocol):
    """Anything that maps an observation to a discrete action."""

    name: str

    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        """Return an action in ``{0, 1, 2}`` for the given observation."""
        ...


class HoldAgent:
    """Always holds. Useful as a trivial control."""

    name = "hold"

    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        return int(Action.HOLD)


class RandomAgent:
    """Samples actions from a seeded RNG. Useful for smoke tests."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        return int(self._rng.integers(0, 3))

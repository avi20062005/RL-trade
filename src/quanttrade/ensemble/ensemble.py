"""Ensemble of agents.

Combines several agents by a weighted vote. To avoid the legacy system's bias
(ties defaulting to HOLD), votes are accumulated as weighted continuous scores
per action; the chosen action is the argmax of those scores. Weights can be
derived from validation Sharpe ratios via a softmax, which never produces zero
or negative weights and degrades gracefully to uniform when agents are equal.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from quanttrade.agents.base import Agent
from quanttrade.env.portfolio import Action
from quanttrade.utils.exceptions import QuantTradeError

_N_ACTIONS = len(Action)


class EnsembleAgent:
    """A weighted ensemble that itself satisfies the Agent protocol."""

    name = "ensemble"

    def __init__(self, agents: Sequence[Agent], weights: Sequence[float] | None = None) -> None:
        if not agents:
            raise QuantTradeError("ensemble requires at least one agent")
        self._agents = list(agents)
        if weights is None:
            weights = [1.0 / len(self._agents)] * len(self._agents)
        if len(weights) != len(self._agents):
            raise QuantTradeError("weights length must match number of agents")
        w = np.asarray(weights, dtype=np.float64)
        if np.any(w < 0):
            raise QuantTradeError("weights must be non-negative")
        total = w.sum()
        if total <= 0:
            raise QuantTradeError("weights must sum to a positive value")
        self._weights: np.ndarray = np.asarray(w / total, dtype=np.float64)

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        scores = np.zeros(_N_ACTIONS, dtype=np.float64)
        for agent, weight in zip(self._agents, self._weights, strict=True):
            action = agent.act(observation, deterministic=deterministic)
            scores[action] += weight
        return int(np.argmax(scores))


def weights_from_sharpe(sharpes: Sequence[float], *, temperature: float = 1.0) -> np.ndarray:
    """Convert validation Sharpe ratios to ensemble weights via a softmax."""
    if temperature <= 0:
        raise QuantTradeError("temperature must be positive")
    arr = np.asarray(sharpes, dtype=np.float64)
    if arr.size == 0:
        raise QuantTradeError("need at least one Sharpe value")
    scaled = (arr - arr.max()) / temperature  # subtract max for numerical stability
    exp = np.exp(scaled)
    return np.asarray(exp / exp.sum(), dtype=np.float64)

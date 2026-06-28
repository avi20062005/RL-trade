"""Stable-Baselines3 agent adapter.

Wraps PPO / DQN / A2C behind the :class:`Agent` protocol. SB3 and torch are
imported lazily, so this module is only required when the ``agents`` extra is
installed. Models are seeded for reproducibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from quanttrade.config import TrainConfig
from quanttrade.env.trading_env import TradingEnv
from quanttrade.utils.exceptions import QuantTradeError
from quanttrade.utils.logging import get_logger

logger = get_logger(__name__)

_ALGOS = ("ppo", "dqn", "a2c")


def _algo_class(algo: str) -> Any:
    from stable_baselines3 import A2C, DQN, PPO

    mapping = {"ppo": PPO, "dqn": DQN, "a2c": A2C}
    if algo not in mapping:
        raise QuantTradeError(f"unknown algo {algo!r}; choose from {_ALGOS}")
    return mapping[algo]


class SB3Agent:
    """Adapts a trained SB3 model to the project's Agent protocol."""

    def __init__(self, model: Any, name: str) -> None:
        self._model = model
        self.name = name

    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        action, _ = self._model.predict(observation, deterministic=deterministic)
        return int(np.asarray(action).item())

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))

    @classmethod
    def load(cls, algo: str, path: str | Path) -> SB3Agent:
        model = _algo_class(algo).load(str(path))
        return cls(model, name=algo)


def train_sb3_agent(algo: str, env: TradingEnv, config: TrainConfig) -> SB3Agent:
    """Train an SB3 agent on the given environment and return the wrapper."""
    policy = "MlpPolicy"
    algo_cls = _algo_class(algo)
    kwargs: dict[str, Any] = {
        "policy": policy,
        "env": env,
        "seed": config.seed,
        "gamma": config.gamma,
        "learning_rate": config.learning_rate,
        "verbose": 0,
    }
    if algo in ("ppo", "a2c"):
        kwargs["device"] = "cpu"  # MLP policies are faster on CPU in SB3
    model = algo_cls(**kwargs)
    logger.info("Training %s for %d timesteps", algo, config.total_timesteps)
    model.learn(total_timesteps=config.total_timesteps)
    return SB3Agent(model, name=algo)

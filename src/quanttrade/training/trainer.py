"""Training orchestration.

Ties together dataset preparation, environment construction, agent training and
artifact persistence. Heavy agents are imported lazily so the orchestration is
importable without the ML extras.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quanttrade.config import AppConfig
from quanttrade.env.trading_env import TradingEnv
from quanttrade.training.datasets import DatasetBundle, SplitData, prepare_datasets
from quanttrade.utils.exceptions import QuantTradeError
from quanttrade.utils.logging import get_logger
from quanttrade.utils.seeding import set_global_seeds

logger = get_logger(__name__)

_SB3_ALGOS = {"ppo", "dqn", "a2c"}


def build_env(split: SplitData, config: AppConfig) -> TradingEnv:
    """Construct a trading environment for a prepared split."""
    return TradingEnv(split.features, split.prices, config.env)


class Trainer:
    """Trains a single agent and persists the model and the normalizer."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._models_dir = Path(config.train.models_dir)

    def prepare(self, ohlcv: pd.DataFrame) -> DatasetBundle:
        """Build leak-free datasets from raw OHLCV."""
        set_global_seeds(self._config.train.seed)
        return prepare_datasets(ohlcv, self._config)

    def train(self, algo: str, bundle: DatasetBundle) -> Path:
        """Train ``algo`` on the train split and persist artifacts.

        Returns the path to the saved model.
        """
        set_global_seeds(self._config.train.seed)
        env = build_env(bundle.train, self._config)
        self._models_dir.mkdir(parents=True, exist_ok=True)

        norm_path = self._models_dir / "normalizer.json"
        bundle.normalizer.save(norm_path)

        if algo in _SB3_ALGOS:
            from quanttrade.agents.sb3_agent import train_sb3_agent

            agent = train_sb3_agent(algo, env, self._config.train)
            model_path = self._models_dir / f"{algo}.zip"
            agent.save(model_path)
        elif algo == "ddqn":
            from quanttrade.agents.ddqn import DoubleDQNAgent

            state_dim = int(env.observation_space.shape[0])  # type: ignore[index]
            ddqn = DoubleDQNAgent(state_dim, self._config.train)
            ddqn.learn(env, self._config.train.total_timesteps)
            model_path = self._models_dir / "ddqn.pt"
            ddqn.save(model_path)
        else:
            raise QuantTradeError(f"unknown algo {algo!r}")

        logger.info("Saved %s model to %s", algo, model_path)
        return model_path

"""Inference.

Turns a recent OHLCV window into a single trading recommendation. The predictor
loads the **train-fit** normalizer and applies it unchanged, and assembles the
observation with the same shared helper the environment uses. This eliminates
the train/serve skew that plagued the legacy predictor (which re-derived
normalization from each live window).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quanttrade.agents.base import Agent
from quanttrade.config import AppConfig
from quanttrade.env.observation import assemble_observation
from quanttrade.env.portfolio import Action
from quanttrade.features.normalizer import FeatureNormalizer
from quanttrade.features.pipeline import FeaturePipeline
from quanttrade.utils.exceptions import QuantTradeError
from quanttrade.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Prediction:
    """A single inference result."""

    action: str
    timestamp: pd.Timestamp
    price: float


class Predictor:
    """Generates trading recommendations from recent market data."""

    def __init__(
        self,
        agent: Agent,
        normalizer: FeatureNormalizer,
        pipeline: FeaturePipeline,
    ) -> None:
        if not normalizer.is_fitted:
            raise QuantTradeError("normalizer must be fitted/loaded before inference")
        self._agent = agent
        self._normalizer = normalizer
        self._pipeline = pipeline

    def predict(
        self, ohlcv: pd.DataFrame, *, exposure: float = 0.0, unrealized_pnl_pct: float = 0.0
    ) -> Prediction:
        """Recommend an action for the most recent bar in ``ohlcv``."""
        feature_set = self._pipeline.transform(ohlcv)
        normalized = self._normalizer.transform(feature_set.features)
        last_row = normalized[-1]
        obs = assemble_observation(last_row, exposure, unrealized_pnl_pct)

        action = self._agent.act(obs, deterministic=True)
        label = Action(action).name
        timestamp = feature_set.price.index[-1]
        price = float(feature_set.price.iloc[-1])
        logger.info("Prediction %s @ %s (price=%.2f)", label, timestamp, price)
        return Prediction(action=label, timestamp=timestamp, price=price)

    @classmethod
    def from_artifacts(cls, agent: Agent, models_dir: str | Path, config: AppConfig) -> Predictor:
        """Build a predictor, loading the normalizer saved during training."""
        norm_path = Path(models_dir) / "normalizer.json"
        if not norm_path.exists():
            raise QuantTradeError(f"normalizer not found at {norm_path}")
        normalizer = FeatureNormalizer.load(norm_path)
        pipeline = FeaturePipeline(config.features)
        return cls(agent, normalizer, pipeline)

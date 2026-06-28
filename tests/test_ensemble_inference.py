"""Tests for the ensemble and the inference predictor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quanttrade.agents.base import Agent, HoldAgent
from quanttrade.config import FeatureConfig
from quanttrade.ensemble.ensemble import EnsembleAgent, weights_from_sharpe
from quanttrade.env.portfolio import Action
from quanttrade.features.normalizer import FeatureNormalizer
from quanttrade.features.pipeline import FeaturePipeline
from quanttrade.inference.predictor import Predictor
from quanttrade.utils.exceptions import QuantTradeError


class _FixedAgent:
    def __init__(self, action: int, name: str) -> None:
        self._action = action
        self.name = name

    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        return self._action


def test_ensemble_weighted_majority() -> None:
    agents: list[Agent] = [_FixedAgent(Action.BUY, "a"), _FixedAgent(Action.BUY, "b"),
                           _FixedAgent(Action.SELL, "c")]
    ens = EnsembleAgent(agents)  # equal weights -> BUY wins 2:1
    assert ens.act(np.zeros(5, dtype=np.float32)) == int(Action.BUY)


def test_ensemble_respects_weights() -> None:
    agents: list[Agent] = [_FixedAgent(Action.BUY, "a"), _FixedAgent(Action.SELL, "b")]
    ens = EnsembleAgent(agents, weights=[0.2, 0.8])  # SELL is weighted higher
    assert ens.act(np.zeros(3, dtype=np.float32)) == int(Action.SELL)


def test_ensemble_rejects_bad_weights() -> None:
    with pytest.raises(QuantTradeError):
        EnsembleAgent([HoldAgent()], weights=[0.5, 0.5])
    with pytest.raises(QuantTradeError):
        EnsembleAgent([])


def test_weights_from_sharpe_softmax() -> None:
    w = weights_from_sharpe([2.0, 1.0, 0.0])
    assert abs(w.sum() - 1.0) < 1e-9
    assert w[0] > w[1] > w[2]  # higher Sharpe -> higher weight
    # Equal Sharpes -> uniform weights.
    eq = weights_from_sharpe([1.0, 1.0, 1.0])
    np.testing.assert_allclose(eq, [1 / 3, 1 / 3, 1 / 3])


def _ohlcv(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(5)
    close = 24_000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    vol = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


def test_predictor_uses_loaded_normalizer(tmp_path: Path) -> None:
    pipe = FeaturePipeline(FeatureConfig())
    feats = pipe.transform(_ohlcv()).features
    norm = FeatureNormalizer().fit(feats.iloc[:120])
    norm_path = tmp_path / "normalizer.json"
    norm.save(norm_path)

    loaded = FeatureNormalizer.load(norm_path)
    predictor = Predictor(HoldAgent(), loaded, pipe)
    pred = predictor.predict(_ohlcv())
    assert pred.action in {a.name for a in Action}
    assert pred.price > 0


def test_predictor_requires_fitted_normalizer() -> None:
    with pytest.raises(QuantTradeError):
        Predictor(HoldAgent(), FeatureNormalizer(), FeaturePipeline(FeatureConfig()))

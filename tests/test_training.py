"""Tests for training orchestration.

The dataset tests run everywhere. The agent smoke tests are skipped unless the
optional ML stack (torch / stable-baselines3) is installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quanttrade.config import AppConfig
from quanttrade.training.datasets import prepare_datasets
from quanttrade.training.trainer import Trainer, build_env


def _ohlcv(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    close = 24_000 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    vol = rng.integers(1_000, 9_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


def test_splits_are_chronological_and_disjoint() -> None:
    bundle = prepare_datasets(_ohlcv(), AppConfig())
    assert bundle.train.index.max() <= bundle.validation.index.min()
    assert bundle.validation.index.max() <= bundle.test.index.min()
    total = len(bundle.train.prices) + len(bundle.validation.prices) + len(bundle.test.prices)
    assert total > 0


def test_normalizer_fit_on_train_only() -> None:
    bundle = prepare_datasets(_ohlcv(), AppConfig())
    # Train features should be ~standardized; test features generally are not.
    assert abs(float(bundle.train.features.mean())) < 0.3
    assert bundle.test.features.shape[1] == len(bundle.feature_names)


def test_build_env_matches_observation_dim() -> None:
    cfg = AppConfig()
    bundle = prepare_datasets(_ohlcv(), cfg)
    env = build_env(bundle.train, cfg)
    expected = len(bundle.feature_names) + 2  # +2 position features
    assert env.observation_space.shape == (expected,)


def test_trainer_prepare_is_reproducible() -> None:
    cfg = AppConfig()
    b1 = Trainer(cfg).prepare(_ohlcv())
    b2 = Trainer(cfg).prepare(_ohlcv())
    np.testing.assert_array_equal(b1.train.features, b2.train.features)


# --------------------------------------------------------------- optional ML

def test_ddqn_smoke() -> None:
    pytest.importorskip("torch")
    from quanttrade.agents.ddqn import DoubleDQNAgent

    cfg = AppConfig.from_dict({"train": {"total_timesteps": 256, "batch_size": 16}})
    bundle = prepare_datasets(_ohlcv(), cfg)
    env = build_env(bundle.train, cfg)
    state_dim = int(env.observation_space.shape[0])
    agent = DoubleDQNAgent(state_dim, cfg.train, buffer_size=500)
    agent.learn(env, cfg.train.total_timesteps)
    obs, _ = env.reset(seed=0)
    assert agent.act(obs) in (0, 1, 2)


def test_sb3_smoke() -> None:
    pytest.importorskip("stable_baselines3")
    from quanttrade.agents.sb3_agent import train_sb3_agent

    cfg = AppConfig.from_dict({"train": {"total_timesteps": 256}})
    bundle = prepare_datasets(_ohlcv(), cfg)
    env = build_env(bundle.train, cfg)
    agent = train_sb3_agent("ppo", env, cfg.train)
    obs, _ = env.reset(seed=0)
    assert agent.act(obs) in (0, 1, 2)

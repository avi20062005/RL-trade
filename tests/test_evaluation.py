"""Tests for evaluation metrics and the backtester."""

from __future__ import annotations

import math

import numpy as np

from quanttrade.agents.base import HoldAgent, RandomAgent
from quanttrade.config import EnvConfig, EvalConfig
from quanttrade.evaluation.backtester import Backtester, walk_forward_splits
from quanttrade.evaluation.metrics import compute_metrics


def test_flat_equity_yields_zero_sharpe_not_explosion() -> None:
    equity = np.full(100, 100_000.0)
    m = compute_metrics(equity, [], bars_per_year=252.0)
    assert m.sharpe == 0.0
    assert m.max_drawdown == 0.0
    assert m.total_return == 0.0


def test_total_return_and_cagr() -> None:
    equity = np.array([100.0, 110.0])  # +10% over 1 bar
    m = compute_metrics(equity, [], bars_per_year=1.0)  # 1 bar/year -> 1 year
    assert m.total_return == math.isclose(0.10, m.total_return) or abs(m.total_return - 0.10) < 1e-9
    assert abs(m.cagr - 0.10) < 1e-9


def test_max_drawdown_is_negative_on_decline() -> None:
    equity = np.array([100.0, 120.0, 60.0, 90.0])
    m = compute_metrics(equity, [], bars_per_year=252.0)
    # peak 120 -> trough 60 = -50%
    assert abs(m.max_drawdown - (-0.5)) < 1e-9


def test_win_rate_and_profit_factor_from_trades() -> None:
    equity = np.array([100.0, 101.0, 102.0])
    m = compute_metrics(equity, [10.0, -5.0, 20.0, -5.0], bars_per_year=252.0)
    assert m.trade_count == 4
    assert abs(m.win_rate - 0.5) < 1e-9
    assert abs(m.profit_factor - (30.0 / 10.0)) < 1e-9


def test_annualization_scales_with_bars_per_year() -> None:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, 1000)
    equity = 100_000.0 * np.cumprod(1 + rets)
    daily = compute_metrics(equity, [], bars_per_year=252.0).annual_volatility
    intraday = compute_metrics(equity, [], bars_per_year=252.0 * 75).annual_volatility
    # More bars/year -> larger annualization factor.
    assert intraday > daily


def _toy_window(n: int = 120):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(2)
    feats = rng.normal(0, 1, size=(n, 5)).astype(np.float32)
    prices = 24_000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.005, n)))
    return feats, prices


def test_backtester_runs_agent_and_benchmark() -> None:
    feats, prices = _toy_window()
    bt = Backtester(feats, prices, EnvConfig(), EvalConfig(), bars_per_year=252.0)
    hold = bt.run_agent(HoldAgent())
    assert hold.metrics.trade_count == 0
    rand = bt.run_agent(RandomAgent(seed=1))
    assert rand.equity_curve.shape[0] == len(prices)
    bh = bt.run_buy_and_hold()
    assert bh.metrics.final_equity > 0


def test_walk_forward_splits_are_chronological() -> None:
    splits = walk_forward_splits(1000, n_splits=4, expanding=True)
    assert len(splits) == 4
    for s in splits:
        assert s.train.stop == s.test.start  # no overlap, test strictly after train
        assert s.train.start == 0  # expanding window starts at 0
    # rolling window does not always start at 0
    rolling = walk_forward_splits(1000, n_splits=4, expanding=False)
    assert any(s.train.start > 0 for s in rolling)

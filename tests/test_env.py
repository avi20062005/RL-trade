"""Tests for portfolio accounting and the Gymnasium trading environment."""

from __future__ import annotations

import numpy as np
import pytest

from quanttrade.config import EnvConfig
from quanttrade.env.portfolio import Action, Portfolio
from quanttrade.env.trading_env import TradingEnv
from quanttrade.utils.exceptions import TradingEnvError


def test_buy_then_liquidate_accounting() -> None:
    cfg = EnvConfig(
        initial_cash=100_000.0,
        position_fraction=0.5,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    pf = Portfolio(cfg)
    fill = pf.buy(step=0, mark_price=24_000.0)  # high-priced index is tradable
    assert fill is not None
    # Deployed 50% of equity in notional; fractional units allowed.
    assert pf.units == pytest.approx(50_000.0 / 24_000.0)
    assert pf.cash == pytest.approx(50_000.0)
    assert pf.equity(24_000.0) == pytest.approx(100_000.0)

    close = pf.liquidate(step=1, mark_price=26_400.0)  # +10% move
    assert close is not None
    assert pf.units == 0.0
    # 50k exposure * 10% = 5k profit, zero costs.
    assert close.realized_pnl == pytest.approx(5_000.0, rel=1e-6)
    assert pf.cash == pytest.approx(105_000.0, rel=1e-6)


def test_cash_never_negative_and_leverage_capped() -> None:
    cfg = EnvConfig(initial_cash=10_000.0, position_fraction=1.0, max_leverage=1.0)
    pf = Portfolio(cfg)
    pf.buy(0, 24_000.0)
    pf.buy(1, 24_000.0)  # second buy should be capped by leverage/cash
    assert pf.cash >= -1e-9
    assert pf.exposure(24_000.0) <= 1.0 + 1e-9


def test_costs_reduce_equity() -> None:
    cfg = EnvConfig(transaction_cost_bps=10.0, slippage_bps=5.0, position_fraction=0.5)
    pf = Portfolio(cfg)
    pf.buy(0, 100.0)
    # A buy then immediate liquidation at the same mark must lose money to costs.
    pf.liquidate(1, 100.0)
    assert pf.equity(100.0) < cfg.initial_cash


def _toy_env(n: int = 100, *, trend: float = 0.0) -> TradingEnv:
    rng = np.random.default_rng(0)
    feats = rng.normal(0, 1, size=(n, 5)).astype(np.float32)
    prices = 24_000.0 * np.exp(np.cumsum(np.full(n, trend) + rng.normal(0, 0.001, n)))
    return TradingEnv(feats, prices, EnvConfig())


def test_env_reset_and_step_shapes() -> None:
    env = _toy_env()
    obs, _info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    obs2, reward, term, trunc, _info2 = env.step(Action.BUY)
    assert obs2.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert term is False and trunc is False


def test_env_runs_to_termination() -> None:
    env = _toy_env(50)
    env.reset(seed=1)
    steps = 0
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = env.step(Action.HOLD)
        steps += 1
    assert steps == 49  # n - 1 transitions
    assert len(env.equity_curve) == 50


def test_env_is_deterministic_given_seed() -> None:
    def rollout() -> float:
        env = _toy_env(60, trend=0.001)
        env.reset(seed=7)
        total = 0.0
        terminated = False
        actions = np.random.default_rng(7).integers(0, 3, size=200)
        i = 0
        while not terminated:
            _, r, terminated, _, _ = env.step(int(actions[i]))
            total += r
            i += 1
        return total

    assert rollout() == pytest.approx(rollout())


def test_stop_loss_forces_exit() -> None:
    # Sharp downtrend should trigger the stop-loss override after a BUY.
    n = 40
    feats = np.zeros((n, 4), dtype=np.float32)
    prices = 24_000.0 * np.exp(np.cumsum(np.full(n, -0.02)))  # -2%/bar
    env = TradingEnv(feats, prices, EnvConfig(stop_loss_pct=0.05, position_fraction=0.5))
    env.reset(seed=0)
    env.step(Action.BUY)
    forced_exit = False
    for _ in range(10):
        _, _, _, _, info = env.step(Action.HOLD)
        if info["action"] == "SELL":
            forced_exit = True
            break
    assert forced_exit


def test_env_rejects_bad_inputs() -> None:
    with pytest.raises(TradingEnvError):
        TradingEnv(np.zeros((5, 3), dtype=np.float32), np.zeros(4), EnvConfig())
    with pytest.raises(TradingEnvError):
        TradingEnv(np.zeros((1, 3), dtype=np.float32), np.array([100.0]), EnvConfig())

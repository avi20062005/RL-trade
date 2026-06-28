"""Backtesting.

Runs a deterministic rollout of an agent through :class:`TradingEnv`, computes
metrics, and provides a buy-and-hold benchmark and chronological walk-forward
splits for honest, leak-free evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quanttrade.agents.base import Agent
from quanttrade.config import EnvConfig, EvalConfig
from quanttrade.env.portfolio import Action
from quanttrade.env.trading_env import TradingEnv
from quanttrade.evaluation.metrics import PerformanceMetrics, compute_metrics
from quanttrade.utils.exceptions import QuantTradeError
from quanttrade.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Outcome of a single backtest."""

    label: str
    metrics: PerformanceMetrics
    equity_curve: np.ndarray


@dataclass(frozen=True, slots=True)
class Split:
    """A chronological train/test index window (half-open ranges)."""

    train: range
    test: range


class Backtester:
    """Evaluates agents and benchmarks on a fixed feature/price window."""

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        env_config: EnvConfig,
        eval_config: EvalConfig,
        bars_per_year: float,
    ) -> None:
        self._features = features
        self._prices = prices
        self._env_config = env_config
        self._eval_config = eval_config
        self._bars_per_year = bars_per_year

    def run_agent(self, agent: Agent, *, label: str | None = None) -> BacktestResult:
        """Roll an agent deterministically through the environment."""
        env = TradingEnv(self._features, self._prices, self._env_config)
        obs, _ = env.reset(seed=0)
        exposures: list[float] = []
        terminated = False
        while not terminated:
            action = agent.act(obs, deterministic=True)
            obs, _, terminated, _, info = env.step(action)
            exposures.append(float(info["exposure"]))

        pnls = [f.realized_pnl for f in env.portfolio.closed_trades]
        metrics = compute_metrics(
            env.equity_curve,
            pnls,
            bars_per_year=self._bars_per_year,
            risk_free_rate_annual=self._eval_config.risk_free_rate_annual,
            exposure_curve=exposures,
        )
        result = BacktestResult(label or agent.name, metrics, env.equity_curve)
        logger.info(
            "Backtest [%s] return=%.2f%% sharpe=%.2f maxDD=%.2f%% trades=%d",
            result.label,
            metrics.total_return * 100,
            metrics.sharpe,
            metrics.max_drawdown * 100,
            metrics.trade_count,
        )
        return result

    def run_buy_and_hold(self) -> BacktestResult:
        """Fully invested benchmark: buy at the first bar, hold to the last."""
        prices = self._prices
        initial = self._env_config.initial_cash
        equity = initial * (prices / prices[0])
        metrics = compute_metrics(
            equity,
            closed_trade_pnls=[float(equity[-1] - equity[0])],
            bars_per_year=self._bars_per_year,
            risk_free_rate_annual=self._eval_config.risk_free_rate_annual,
            exposure_curve=np.ones(len(equity)),
        )
        return BacktestResult("buy_and_hold", metrics, equity)


def walk_forward_splits(
    n: int, n_splits: int, *, test_fraction: float = 0.2, expanding: bool = True
) -> list[Split]:
    """Generate chronological walk-forward splits over ``n`` ordered samples.

    Args:
        n: Number of time-ordered samples.
        n_splits: Number of train/test folds.
        test_fraction: Test-window size as a fraction of one block.
        expanding: If True the train window grows; otherwise it slides.
    """
    if n_splits < 1:
        raise QuantTradeError("n_splits must be >= 1")
    block = n // (n_splits + 1)
    if block < 2:
        raise QuantTradeError("not enough samples for the requested number of splits")
    test_len = max(1, int(block * test_fraction) or block)

    splits: list[Split] = []
    for i in range(1, n_splits + 1):
        test_start = i * block
        test_end = min(test_start + test_len, n)
        if test_start >= n:
            break
        train_start = 0 if expanding else max(0, test_start - block)
        splits.append(Split(train=range(train_start, test_start), test=range(test_start, test_end)))
    return splits


__all__ = [
    "Action",
    "BacktestResult",
    "Backtester",
    "Split",
    "walk_forward_splits",
]

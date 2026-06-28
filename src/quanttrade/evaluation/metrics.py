"""Performance metrics.

All annualization uses ``bars_per_year`` derived from the bar interval, so the
same code is correct for daily or 5-minute data. Degenerate inputs (flat equity,
zero variance, no trades) return zeros rather than exploding — the legacy system
reported a Sharpe of -3.6e6 by dividing a constant excess return by ~0.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Container for backtest performance statistics."""

    total_return: float
    cagr: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    trade_count: int
    exposure: float
    final_equity: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / np.where(peak > 0, peak, 1.0)
    return float(drawdowns.min())


def compute_metrics(
    equity_curve: Sequence[float] | np.ndarray,
    closed_trade_pnls: Sequence[float],
    *,
    bars_per_year: float,
    risk_free_rate_annual: float = 0.0,
    exposure_curve: Sequence[float] | np.ndarray | None = None,
) -> PerformanceMetrics:
    """Compute performance metrics from an equity curve and closed-trade PnLs."""
    equity = np.asarray(equity_curve, dtype=np.float64)
    n = len(equity)
    if n < 2 or equity[0] <= 0:
        return _empty_metrics(final_equity=float(equity[-1]) if n else 0.0)

    returns = np.diff(equity) / equity[:-1]
    rf_per_bar = (1.0 + risk_free_rate_annual) ** (1.0 / bars_per_year) - 1.0
    excess = returns - rf_per_bar

    sqrt_ppy = float(np.sqrt(bars_per_year))
    ret_std = float(returns.std())
    excess_std = float(excess.std())

    sharpe = (excess.mean() / excess_std) * sqrt_ppy if excess_std > _EPS else 0.0
    downside = excess[excess < 0.0]
    downside_std = float(downside.std()) if downside.size > 0 else 0.0
    sortino = (excess.mean() / downside_std) * sqrt_ppy if downside_std > _EPS else 0.0

    total_return = float(equity[-1] / equity[0] - 1.0)
    years = (n - 1) / bars_per_year
    cagr = float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    annual_vol = ret_std * sqrt_ppy
    max_dd = _max_drawdown(equity)
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > _EPS else 0.0

    pnls = np.asarray(closed_trade_pnls, dtype=np.float64)
    trade_count = int(pnls.size)
    if trade_count > 0:
        wins = pnls[pnls > 0.0]
        losses = pnls[pnls < 0.0]
        win_rate = float(wins.size / trade_count)
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        profit_factor = gross_profit / gross_loss if gross_loss > _EPS else float("inf")
        avg_trade = float(pnls.mean())
    else:
        win_rate = profit_factor = avg_trade = 0.0

    if exposure_curve is not None:
        exposure = float(np.mean(np.asarray(exposure_curve, dtype=np.float64) > _EPS))
    else:
        exposure = 0.0

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        annual_return=cagr,
        annual_volatility=annual_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_pnl=avg_trade,
        trade_count=trade_count,
        exposure=exposure,
        final_equity=float(equity[-1]),
    )


def _empty_metrics(final_equity: float) -> PerformanceMetrics:
    return PerformanceMetrics(
        total_return=0.0,
        cagr=0.0,
        annual_return=0.0,
        annual_volatility=0.0,
        sharpe=0.0,
        sortino=0.0,
        max_drawdown=0.0,
        calmar=0.0,
        win_rate=0.0,
        profit_factor=0.0,
        avg_trade_pnl=0.0,
        trade_count=0,
        exposure=0.0,
        final_equity=final_equity,
    )

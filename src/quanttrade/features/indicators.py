"""Causal technical indicators.

Every function uses only the current bar and prior bars (via ``diff``, ``ewm``
and ``rolling`` with no negative shifts), so there is no look-ahead. Indicators
are returned in *scale-free* form (ratios, z-scores, percentages) so the model
does not key off the absolute price level, which is non-stationary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12


def log_return(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Log return over ``horizon`` bars: ``log(P_t / P_{t-horizon})``."""
    return pd.Series(np.log(close / close.shift(horizon)), index=close.index)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100] using an exponentially weighted average."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / (avg_loss + _EPS)
    return (100.0 - 100.0 / (1.0 + rs)).clip(0.0, 100.0)


def atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range expressed as a fraction of price (scale-free)."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    return atr / (close + _EPS)


def macd_hist_norm(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """MACD histogram normalized by price (scale-free)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return (macd_line - signal_line) / (close + _EPS)


def bollinger_width(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger band width relative to the moving average (scale-free)."""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (2.0 * num_std * std) / (mean + _EPS)


def bollinger_position(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Standardized position of price within its Bollinger band."""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - mean) / (num_std * std + _EPS)


def realized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of 1-bar log returns."""
    return log_return(close, 1).rolling(window).std()


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """Z-score of log-volume against its rolling mean/std (scale-free)."""
    log_vol = pd.Series(np.log1p(volume.clip(lower=0.0)), index=volume.index)
    mean = log_vol.rolling(window).mean()
    std = log_vol.rolling(window).std()
    return (log_vol - mean) / (std + _EPS)

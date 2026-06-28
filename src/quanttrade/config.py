"""Typed, validated configuration.

Design notes
------------
* Every config object is a *frozen* dataclass validated at construction, so an
  invalid configuration fails immediately and loudly (``ConfigError``) rather
  than producing subtle runtime bugs deep in training.
* There is **no global singleton**. The application constructs one
  :class:`AppConfig` and passes it explicitly. This keeps components pure and
  testable and avoids hidden shared mutable state.
* The bar interval drives the *annualization factor* used everywhere in
  evaluation, which is the correct way to annualize sub-daily returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any

from quanttrade.utils.exceptions import ConfigError

# Mapping of supported interval strings to their length in minutes.
# ``1d`` is represented as a full session so that ``bars_per_day == 1``.
_INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "1h": 60,
    "1d": 0,  # sentinel; handled explicitly as one bar per day
}


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Market-data and calendar configuration."""

    symbol: str = "^NSEI"
    interval: str = "1d"
    session_minutes: int = 375  # NSE regular session 09:15-15:30
    trading_days_per_year: int = 252
    cache_dir: str = "artifacts/cache"
    max_gap_bars: int = 5  # consecutive missing bars tolerated before erroring

    def __post_init__(self) -> None:
        if self.interval not in _INTERVAL_MINUTES:
            raise ConfigError(
                f"Unsupported interval {self.interval!r}; "
                f"choose from {sorted(_INTERVAL_MINUTES)}"
            )
        if self.session_minutes <= 0:
            raise ConfigError("session_minutes must be positive")
        if self.trading_days_per_year <= 0:
            raise ConfigError("trading_days_per_year must be positive")

    @property
    def bars_per_day(self) -> float:
        """Number of bars in one trading day for this interval."""
        if self.interval == "1d":
            return 1.0
        return self.session_minutes / _INTERVAL_MINUTES[self.interval]

    @property
    def bars_per_year(self) -> float:
        """Annualization factor: bars per year for this interval.

        Used to scale per-bar returns to annual figures (Sharpe, volatility,
        CAGR). Annualizing 5-minute returns with ``sqrt(252)`` — as the legacy
        system did — understates volatility by ``sqrt(bars_per_day)``.
        """
        return self.bars_per_day * self.trading_days_per_year


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Windows for causal technical indicators."""

    return_horizons: tuple[int, ...] = (1, 5, 15)
    rsi_window: int = 14
    atr_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0
    vol_window: int = 20

    def __post_init__(self) -> None:
        positive = {
            "rsi_window": self.rsi_window,
            "atr_window": self.atr_window,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "bb_window": self.bb_window,
            "vol_window": self.vol_window,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ConfigError(f"{name} must be positive, got {value}")
        if self.macd_fast >= self.macd_slow:
            raise ConfigError("macd_fast must be strictly less than macd_slow")
        if self.bb_std <= 0:
            raise ConfigError("bb_std must be positive")
        if any(h <= 0 for h in self.return_horizons):
            raise ConfigError("return_horizons must all be positive")


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Trading-environment economics and risk parameters."""

    initial_cash: float = 100_000.0
    position_fraction: float = 0.20  # fraction of equity deployed per BUY
    max_leverage: float = 1.0  # 1.0 = long-only, no leverage
    transaction_cost_bps: float = 5.0  # round-trip cost is paid on each side
    slippage_bps: float = 2.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    reward_scaling: float = 1.0  # reward = scaling * log-growth of equity

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ConfigError("initial_cash must be positive")
        if not 0.0 < self.position_fraction <= 1.0:
            raise ConfigError("position_fraction must be in (0, 1]")
        if self.max_leverage < 1.0:
            raise ConfigError("max_leverage must be >= 1.0")
        for name in ("transaction_cost_bps", "slippage_bps"):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} must be non-negative")
        for name in ("stop_loss_pct", "take_profit_pct"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Chronological train/validation/test split fractions."""

    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-9:
            raise ConfigError(f"split fractions must sum to 1.0, got {total}")
        if min(self.train, self.validation, self.test) <= 0:
            raise ConfigError("each split fraction must be positive")


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Evaluation parameters."""

    risk_free_rate_annual: float = 0.06
    split: SplitConfig = field(default_factory=SplitConfig)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Training hyperparameters shared across agents."""

    seed: int = 42
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 64
    models_dir: str = "artifacts/models"

    def __post_init__(self) -> None:
        if self.total_timesteps <= 0:
            raise ConfigError("total_timesteps must be positive")
        if not 0.0 < self.gamma <= 1.0:
            raise ConfigError("gamma must be in (0, 1]")
        if self.learning_rate <= 0:
            raise ConfigError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ConfigError("batch_size must be positive")


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level configuration composed of independent sections."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, overrides: dict[str, Any]) -> AppConfig:
        """Build a config, overriding defaults from a (possibly nested) dict.

        Unknown keys raise :class:`ConfigError` so typos cannot silently no-op.
        """
        result = _build_dataclass(cls, overrides)
        assert isinstance(result, cls)
        return result


def _build_dataclass(klass: type, overrides: dict[str, Any]) -> Any:
    """Recursively construct a frozen dataclass from a dict of overrides."""
    if not isinstance(overrides, dict):
        raise ConfigError(f"Expected a mapping for {klass.__name__}, got {type(overrides)}")

    field_map = {f.name: f for f in fields(klass)}
    unknown = set(overrides) - set(field_map)
    if unknown:
        raise ConfigError(f"Unknown config keys for {klass.__name__}: {sorted(unknown)}")

    instance = klass()
    changes: dict[str, Any] = {}
    for key, value in overrides.items():
        current = getattr(instance, key)
        if is_dataclass(type(current)) and isinstance(value, dict):
            changes[key] = _build_dataclass(type(current), value)
        else:
            changes[key] = value
    return replace(instance, **changes)

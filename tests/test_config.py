"""Tests for configuration construction, validation and annualization."""

from __future__ import annotations

import math

import pytest

from quanttrade.config import AppConfig, DataConfig, EnvConfig, SplitConfig
from quanttrade.utils.exceptions import ConfigError


def test_defaults_construct() -> None:
    cfg = AppConfig()
    assert cfg.data.symbol == "^NSEI"
    assert cfg.env.initial_cash > 0


def test_bars_per_year_daily() -> None:
    cfg = DataConfig(interval="1d", trading_days_per_year=252)
    assert cfg.bars_per_day == 1.0
    assert cfg.bars_per_year == 252.0


def test_bars_per_year_intraday() -> None:
    cfg = DataConfig(interval="5m", session_minutes=375, trading_days_per_year=252)
    assert math.isclose(cfg.bars_per_day, 75.0)
    assert math.isclose(cfg.bars_per_year, 75.0 * 252)


def test_invalid_interval_rejected() -> None:
    with pytest.raises(ConfigError):
        DataConfig(interval="7m")


def test_position_fraction_bounds() -> None:
    with pytest.raises(ConfigError):
        EnvConfig(position_fraction=0.0)
    with pytest.raises(ConfigError):
        EnvConfig(position_fraction=1.5)


def test_split_must_sum_to_one() -> None:
    with pytest.raises(ConfigError):
        SplitConfig(train=0.8, validation=0.15, test=0.15)


def test_from_dict_nested_override() -> None:
    cfg = AppConfig.from_dict({"env": {"initial_cash": 5_000.0}, "log_level": "DEBUG"})
    assert cfg.env.initial_cash == 5_000.0
    assert cfg.log_level == "DEBUG"
    # untouched sections keep defaults
    assert cfg.data.symbol == "^NSEI"


def test_from_dict_unknown_key_raises() -> None:
    with pytest.raises(ConfigError):
        AppConfig.from_dict({"env": {"initial_capital": 1.0}})
    with pytest.raises(ConfigError):
        AppConfig.from_dict({"nonexistent": 1})


def test_config_is_frozen() -> None:
    cfg = EnvConfig()
    with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError is a subclass
        cfg.initial_cash = 1.0  # type: ignore[misc]

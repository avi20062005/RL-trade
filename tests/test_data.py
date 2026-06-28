"""Tests for the data layer: schema, cache and loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quanttrade.config import DataConfig
from quanttrade.data.cache import OhlcvCache
from quanttrade.data.loader import DataLoader
from quanttrade.data.schema import validate_ohlcv
from quanttrade.utils.exceptions import DataUnavailableError, SchemaError


def _make_frame(n: int = 50, *, tz: str | None = "UTC") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01 09:15", periods=n, freq="D", tz=tz)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.abs(close) + 1.0
    open_ = close + rng.normal(0, 0.2, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, n))
    vol = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx
    )


def test_validate_ok() -> None:
    out = validate_ohlcv(_make_frame())
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert str(out.index.tz) == "UTC"
    assert out.index.is_monotonic_increasing


def test_validate_requires_tz() -> None:
    with pytest.raises(SchemaError):
        validate_ohlcv(_make_frame(tz=None))


def test_validate_rejects_high_below_low() -> None:
    frame = _make_frame()
    frame.iloc[5, frame.columns.get_loc("High")] = 0.0  # high becomes < low
    with pytest.raises(SchemaError):
        validate_ohlcv(frame)


def test_validate_rejects_negative_prices() -> None:
    frame = _make_frame()
    frame.iloc[3, frame.columns.get_loc("Close")] = -1.0
    with pytest.raises(SchemaError):
        validate_ohlcv(frame)


def test_validate_dedups_and_sorts() -> None:
    frame = _make_frame(10)
    shuffled = pd.concat([frame.iloc[5:], frame.iloc[:5], frame.iloc[:2]])
    out = validate_ohlcv(shuffled)
    assert out.index.is_monotonic_increasing
    assert not out.index.has_duplicates


def test_cache_roundtrip_and_freshness(tmp_path: Path) -> None:
    cache = OhlcvCache(tmp_path)
    frame = validate_ohlcv(_make_frame())
    assert cache.get("^NSEI", "1d", max_age_seconds=10) is None
    cache.put("^NSEI", "1d", frame)
    fresh = cache.get("^NSEI", "1d", max_age_seconds=10_000)
    assert fresh is not None and len(fresh) == len(frame)
    # With a zero TTL the entry is considered stale.
    assert cache.get("^NSEI", "1d", max_age_seconds=-1) is None


class _FakeProvider:
    def __init__(self, frame: pd.DataFrame | None) -> None:
        self._frame = frame

    def fetch(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        if self._frame is None:
            raise DataUnavailableError("simulated outage")
        return self._frame


def test_loader_fetches_validates_and_caches(tmp_path: Path) -> None:
    cfg = DataConfig(symbol="TEST", interval="1d", cache_dir=str(tmp_path))
    loader = DataLoader(cfg, provider=_FakeProvider(_make_frame()), cache=OhlcvCache(tmp_path))
    out = loader.load(period="1y")
    assert "close" in out.columns
    # Second load should be served from cache without the provider.
    out2 = loader.load(period="1y", max_cache_age_seconds=10_000)
    assert len(out2) == len(out)


def test_loader_fails_closed_on_outage(tmp_path: Path) -> None:
    cfg = DataConfig(symbol="TEST", interval="1d", cache_dir=str(tmp_path))
    loader = DataLoader(cfg, provider=_FakeProvider(None), cache=OhlcvCache(tmp_path))
    with pytest.raises(DataUnavailableError):
        loader.load(force_refresh=True)

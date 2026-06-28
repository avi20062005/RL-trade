"""Tests for feature computation and the train-only normalizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quanttrade.config import FeatureConfig
from quanttrade.features.normalizer import FeatureNormalizer
from quanttrade.features.pipeline import FeaturePipeline
from quanttrade.utils.exceptions import FeatureError, NormalizerError


def _ohlcv(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.abs(close) + 5.0
    open_ = close + rng.normal(0, 0.2, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, n))
    vol = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


def test_pipeline_produces_aligned_features() -> None:
    pipe = FeaturePipeline(FeatureConfig())
    result = pipe.transform(_ohlcv())
    assert list(result.features.columns) == pipe.feature_names
    assert result.features.index.equals(result.price.index)
    assert not result.features.isna().to_numpy().any()


def test_features_are_causal() -> None:
    """Changing a future bar must not change an earlier feature row."""
    pipe = FeaturePipeline(FeatureConfig())
    base = _ohlcv()
    perturbed = base.copy()
    perturbed.iloc[-1, perturbed.columns.get_loc("close")] *= 1.5

    f_base = pipe.transform(base).features
    f_pert = pipe.transform(perturbed).features
    common = f_base.index.intersection(f_pert.index)[:-1]  # exclude the changed last row
    pd.testing.assert_frame_equal(f_base.loc[common], f_pert.loc[common])


def test_pipeline_rejects_short_input() -> None:
    pipe = FeaturePipeline(FeatureConfig())
    with pytest.raises(FeatureError):
        pipe.transform(_ohlcv(10))


def test_normalizer_fit_transform_uses_train_stats() -> None:
    pipe = FeaturePipeline(FeatureConfig())
    feats = pipe.transform(_ohlcv()).features
    train, test = feats.iloc[:100], feats.iloc[100:]

    norm = FeatureNormalizer().fit(train)
    out = norm.transform(test)
    assert out.dtype == np.float32
    assert out.shape == (len(test), feats.shape[1])
    # train columns should be roughly standardized
    train_out = norm.transform(train)
    assert abs(float(train_out.mean())) < 0.2


def test_normalizer_requires_fit() -> None:
    pipe = FeaturePipeline(FeatureConfig())
    feats = pipe.transform(_ohlcv()).features
    with pytest.raises(NormalizerError):
        FeatureNormalizer().transform(feats)


def test_normalizer_save_load_roundtrip(tmp_path: Path) -> None:
    pipe = FeaturePipeline(FeatureConfig())
    feats = pipe.transform(_ohlcv()).features
    norm = FeatureNormalizer().fit(feats.iloc[:100])
    path = tmp_path / "norm.json"
    norm.save(path)

    reloaded = FeatureNormalizer.load(path)
    np.testing.assert_allclose(
        norm.transform(feats.iloc[100:]), reloaded.transform(feats.iloc[100:])
    )


def test_normalizer_rejects_column_mismatch() -> None:
    pipe = FeaturePipeline(FeatureConfig())
    feats = pipe.transform(_ohlcv()).features
    norm = FeatureNormalizer().fit(feats)
    with pytest.raises(NormalizerError):
        norm.transform(feats.rename(columns={"rsi": "RSI"}))

"""Dataset preparation.

Builds the train/validation/test matrices used for training and evaluation.
The split is strictly chronological and the normalizer is fit on the training
slice only, then applied unchanged to every split. This is the orchestration
that makes the whole pipeline leak-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quanttrade.config import AppConfig
from quanttrade.features.normalizer import FeatureNormalizer
from quanttrade.features.pipeline import FeaturePipeline
from quanttrade.utils.exceptions import DataError
from quanttrade.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SplitData:
    """Normalized features and aligned prices for one chronological split."""

    name: str
    features: np.ndarray  # float32 [T, F], normalized with train statistics
    prices: np.ndarray  # float64 [T]
    index: pd.DatetimeIndex


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """All splits plus the fitted normalizer and metadata."""

    train: SplitData
    validation: SplitData
    test: SplitData
    normalizer: FeatureNormalizer
    feature_names: list[str]
    bars_per_year: float


def prepare_datasets(ohlcv: pd.DataFrame, config: AppConfig) -> DatasetBundle:
    """Create normalized, leak-free train/val/test splits from OHLCV data."""
    pipeline = FeaturePipeline(config.features)
    feature_set = pipeline.transform(ohlcv)
    features, prices = feature_set.features, feature_set.price

    n = len(prices)
    split = config.evaluation.split
    n_train = int(n * split.train)
    n_val = int(n * split.validation)
    if min(n_train, n_val, n - n_train - n_val) < 2:
        raise DataError(f"not enough rows ({n}) for the configured split")

    bounds = {
        "train": (0, n_train),
        "validation": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }

    normalizer = FeatureNormalizer().fit(features.iloc[bounds["train"][0] : bounds["train"][1]])

    splits: dict[str, SplitData] = {}
    for name, (start, end) in bounds.items():
        feat_slice = features.iloc[start:end]
        splits[name] = SplitData(
            name=name,
            features=normalizer.transform(feat_slice),
            prices=prices.iloc[start:end].to_numpy(dtype=np.float64),
            index=pd.DatetimeIndex(feat_slice.index),
        )
        logger.info("Split %-10s: %d rows", name, end - start)

    return DatasetBundle(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        normalizer=normalizer,
        feature_names=pipeline.feature_names,
        bars_per_year=config.data.bars_per_year,
    )

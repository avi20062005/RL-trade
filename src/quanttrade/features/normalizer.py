"""Feature normalization.

The normalizer is **fit on the training set only**, then applied unchanged to
validation, test and live data. This is the single most important guard against
train/serve skew and look-ahead leakage: every split sees the same fixed
statistics, computed without any knowledge of future bars.

Statistics are persisted as JSON (never pickle) so a saved model and its
normalizer can be reloaded safely and reproducibly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quanttrade.utils.exceptions import NormalizerError
from quanttrade.utils.logging import get_logger

logger = get_logger(__name__)

_EPS = 1e-8


class FeatureNormalizer:
    """Z-score normalizer with persisted, train-only statistics."""

    def __init__(self, clip: float = 5.0) -> None:
        if clip <= 0:
            raise NormalizerError("clip must be positive")
        self._clip = clip
        self._columns: list[str] | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._mean is not None

    def fit(self, train_features: pd.DataFrame) -> FeatureNormalizer:
        """Compute mean/std from the training features only."""
        if train_features.empty:
            raise NormalizerError("Cannot fit on empty features")
        self._columns = list(train_features.columns)
        self._mean = train_features.to_numpy(dtype=np.float64).mean(axis=0)
        self._std = train_features.to_numpy(dtype=np.float64).std(axis=0)
        logger.info(
            "Fitted normalizer on %d rows, %d features", len(train_features), len(self._columns)
        )
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Apply the frozen statistics; returns a float32 matrix.

        Raises:
            NormalizerError: If unfitted or columns do not match the fit set.
        """
        if self._mean is None or self._std is None or self._columns is None:
            raise NormalizerError("Normalizer must be fitted before transform")
        if list(features.columns) != self._columns:
            raise NormalizerError(
                f"Feature columns {list(features.columns)} do not match "
                f"fitted columns {self._columns}"
            )
        values = features.to_numpy(dtype=np.float64)
        normed = (values - self._mean) / (self._std + _EPS)
        clipped = np.clip(normed, -self._clip, self._clip)
        return np.asarray(clipped, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        """Persist statistics as JSON."""
        if self._mean is None or self._std is None or self._columns is None:
            raise NormalizerError("Nothing to save; normalizer is unfitted")
        payload = {
            "clip": self._clip,
            "columns": self._columns,
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> FeatureNormalizer:
        """Reload a normalizer previously saved with :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        norm = cls(clip=float(payload["clip"]))
        norm._columns = list(payload["columns"])
        norm._mean = np.asarray(payload["mean"], dtype=np.float64)
        norm._std = np.asarray(payload["std"], dtype=np.float64)
        return norm

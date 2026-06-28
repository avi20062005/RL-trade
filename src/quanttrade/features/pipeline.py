"""Feature pipeline.

Transforms a validated OHLCV frame into a matrix of causal, scale-free features
plus the aligned execution price. Warmup rows (NaN from indicator windows) are
dropped once, after which the price series and feature matrix share an index.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quanttrade.config import FeatureConfig
from quanttrade.features import indicators as ind
from quanttrade.utils.exceptions import FeatureError
from quanttrade.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """Output of the pipeline: aligned features and execution price."""

    features: pd.DataFrame
    price: pd.Series

    def __post_init__(self) -> None:
        if not self.features.index.equals(self.price.index):
            raise FeatureError("features and price indices are not aligned")


class FeaturePipeline:
    """Builds the model's observation features from OHLCV data."""

    def __init__(self, config: FeatureConfig) -> None:
        self._cfg = config

    @property
    def feature_names(self) -> list[str]:
        cfg = self._cfg
        names = [f"logret_{h}" for h in cfg.return_horizons]
        names += ["rsi", "atr_pct", "macd_hist", "bb_width", "bb_pos", "rvol", "vol_z"]
        return names

    def transform(self, ohlcv: pd.DataFrame) -> FeatureSet:
        """Compute features from a validated OHLCV frame.

        Raises:
            FeatureError: If there are not enough rows after warmup.
        """
        cfg = self._cfg
        close, high, low, volume = (
            ohlcv["close"],
            ohlcv["high"],
            ohlcv["low"],
            ohlcv["volume"],
        )

        cols: dict[str, pd.Series] = {}
        for horizon in cfg.return_horizons:
            cols[f"logret_{horizon}"] = ind.log_return(close, horizon)
        cols["rsi"] = ind.rsi(close, cfg.rsi_window)
        cols["atr_pct"] = ind.atr_pct(high, low, close, cfg.atr_window)
        cols["macd_hist"] = ind.macd_hist_norm(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
        cols["bb_width"] = ind.bollinger_width(close, cfg.bb_window, cfg.bb_std)
        cols["bb_pos"] = ind.bollinger_position(close, cfg.bb_window, cfg.bb_std)
        cols["rvol"] = ind.realized_volatility(close, cfg.vol_window)
        cols["vol_z"] = ind.volume_zscore(volume, cfg.vol_window)

        features = pd.DataFrame(cols, index=ohlcv.index)[self.feature_names]

        combined = features.copy()
        combined["__price__"] = close
        combined = combined.dropna()
        if len(combined) < 2:
            raise FeatureError(f"Too few rows after warmup: {len(combined)}")

        price = combined.pop("__price__")
        dropped = len(ohlcv) - len(combined)
        logger.info(
            "Computed %d features; dropped %d warmup rows", len(self.feature_names), dropped
        )
        return FeatureSet(features=combined, price=price)

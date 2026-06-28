"""Structured exception hierarchy for the framework.

A single base class lets callers catch *all* framework errors with one
``except QuantTradeError`` while still allowing narrow, intentional handling of
specific failure modes. We never raise bare ``Exception``.
"""

from __future__ import annotations


class QuantTradeError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(QuantTradeError):
    """Raised when configuration values are missing or inconsistent."""


class DataError(QuantTradeError):
    """Base class for data-pipeline failures."""


class DataUnavailableError(DataError):
    """Raised when requested market data cannot be obtained.

    The pipeline fails closed: it never fabricates prices to "keep running".
    """


class SchemaError(DataError):
    """Raised when a DataFrame does not satisfy the OHLCV contract."""


class FeatureError(QuantTradeError):
    """Raised when feature computation cannot proceed (e.g. too few rows)."""


class NormalizerError(QuantTradeError):
    """Raised when a normalizer is used before fitting or with unknown columns."""


class TradingEnvError(QuantTradeError):
    """Raised on invalid interaction with the trading environment."""

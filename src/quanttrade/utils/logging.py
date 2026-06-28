"""Logging configuration.

One ``configure_logging`` call sets up the root handlers (console + optional
rotating file). Modules then obtain a child logger via ``get_logger(__name__)``.
This avoids the common bug of attaching duplicate handlers per call and keeps
log files bounded in size.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-32s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_ROOT_NAME = "quanttrade"
_configured = False


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> None:
    """Configure the package root logger. Safe to call more than once.

    Args:
        level: Logging level name (e.g. ``"INFO"``).
        log_file: Optional path for a size-rotated log file.
        max_bytes: Rotate the file once it exceeds this size.
        backup_count: Number of rotated files to retain.
    """
    global _configured
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False

    if _configured:
        return

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the package root namespace."""
    suffix = name.split(".")[-1] if name else "app"
    return logging.getLogger(f"{_ROOT_NAME}.{suffix}")

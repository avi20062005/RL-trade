"""Tests for dashboard model discovery (no Streamlit needed)."""

from __future__ import annotations

from pathlib import Path

from quanttrade.config import AppConfig
from quanttrade.dashboard.models import discover_trained_agents


def test_discovery_empty_when_no_models(tmp_path: Path) -> None:
    config = AppConfig.from_dict({"train": {"models_dir": str(tmp_path / "models")}})
    assert discover_trained_agents(config) == []


def test_discovery_skips_unloadable_files(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    # A bogus checkpoint should be skipped, not crash the page.
    (models / "ppo.zip").write_bytes(b"not a real model")
    config = AppConfig.from_dict({"train": {"models_dir": str(models)}})
    assert discover_trained_agents(config) == []

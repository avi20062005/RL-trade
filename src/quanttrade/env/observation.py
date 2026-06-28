"""Shared observation assembly.

Both the training environment and the inference predictor build observations
through this single function, guaranteeing that what the policy sees at serving
time matches exactly what it saw during training (no train/serve skew).
"""

from __future__ import annotations

import numpy as np

# Number of engineered position features appended to the market features.
POSITION_FEATURES = 2


def assemble_observation(
    market_features: np.ndarray, exposure: float, unrealized_pnl_pct: float
) -> np.ndarray:
    """Concatenate normalized market features with position state."""
    position = np.array([exposure, unrealized_pnl_pct], dtype=np.float32)
    return np.concatenate([market_features.astype(np.float32), position]).astype(np.float32)

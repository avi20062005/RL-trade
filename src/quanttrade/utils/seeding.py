"""Reproducibility helpers.

``set_global_seeds`` seeds Python, NumPy and (if installed) PyTorch from a
single integer so that an entire training run can be reproduced. PyTorch is
imported lazily so the core package does not depend on it.
"""

from __future__ import annotations

import os
import random

import numpy as np

from quanttrade.utils.logging import get_logger

logger = get_logger(__name__)


def set_global_seeds(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed all relevant RNGs.

    Args:
        seed: The master seed.
        deterministic_torch: If True and torch is available, request
            deterministic cuDNN kernels (slower but reproducible).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        logger.debug("torch not installed; skipping torch seeding")
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.debug("Seeded python/numpy/torch with %d", seed)

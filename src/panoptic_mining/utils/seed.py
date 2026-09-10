"""Reproducibility helper: seed every RNG this project touches."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 0) -> None:
    """Seed Python's, numpy's (and, once torch is a hard dependency,
    torch's) RNGs. Kept as a single call site so every experiment config
    can log one seed value and reproduce it."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

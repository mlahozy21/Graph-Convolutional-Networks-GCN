"""Small utilities shared across the package."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic so repeated runs with the same seed are
    # bit-for-bit reproducible (at a small performance cost).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

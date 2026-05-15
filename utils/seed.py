"""
Seed utility for reproducibility across all experiments.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set all random seeds for full reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic ops (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_seeds() -> list:
    """Return the canonical multi-seed list used across all experiments."""
    return [42, 0, 1, 2, 3]

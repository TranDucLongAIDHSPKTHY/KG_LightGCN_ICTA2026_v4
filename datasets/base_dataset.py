"""
datasets/base_dataset.py
─────────────────────────────────────────────────────────────────────────────
Abstract base class for all dataset implementations.
Provides shared I/O, negative sampling, and split-loading utilities.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(ABC):
    """
    Abstract base for CF and KG datasets.

    Subclasses must implement:
        - load()
        - __getitem__() (via torch Dataset mixin)
        - __len__()
    """

    def __init__(self, data_dir: str, split: str = "train", seed: int = 42) -> None:
        """
        Args:
            data_dir: Path to processed dataset directory.
            split:    One of 'train', 'val', 'test'.
            seed:     RNG seed for negative sampling.
        """
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.data_dir = data_dir
        self.split = split
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.n_users: int = 0
        self.n_items: int = 0
        self.user2items: Dict[int, List[int]] = {}
        self.all_train_items: Set[int] = set()

    # ── I/O ──────────────────────────────────────────────────────────────────

    @staticmethod
    def read_interaction_file(path: str) -> Dict[int, List[int]]:
        """Read 'uid item1 item2 …' file → dict."""
        user2items: Dict[int, List[int]] = {}
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                uid = int(parts[0])
                user2items[uid] = [int(x) for x in parts[1:]]
        return user2items

    def _load_split(self, split_name: str) -> Dict[int, List[int]]:
        path = os.path.join(self.data_dir, f"{split_name}.txt")
        if not os.path.exists(path):
            return {}
        return self.read_interaction_file(path)

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def n_interactions(self) -> int:
        return sum(len(v) for v in self.user2items.values())

    def _compute_dimensions(
        self, *splits: Dict[int, List[int]]
    ) -> Tuple[int, int]:
        """Compute n_users and n_items from one or more split dicts."""
        all_users: Set[int] = set()
        all_items: Set[int] = set()
        for d in splits:
            for u, items in d.items():
                all_users.add(u)
                all_items.update(items)
        return max(all_users) + 1, max(all_items) + 1

    # ── Negative sampling ─────────────────────────────────────────────────────

    def sample_negative(self, uid: int, n_neg: int = 1) -> List[int]:
        """
        Uniform negative sampling for a user.
        Excludes all items in train (self.all_train_items[uid]).

        Args:
            uid:   User ID.
            n_neg: Number of negatives to sample.

        Returns:
            List of negative item IDs.
        """
        user_positives = set(self.user2items.get(uid, []))
        negatives: List[int] = []
        while len(negatives) < n_neg:
            neg = self.rng.randint(0, self.n_items)
            if neg not in user_positives:
                negatives.append(neg)
        return negatives

    # ── Abstract ──────────────────────────────────────────────────────────────

    @abstractmethod
    def load(self) -> None:
        """Load and prepare data from disk."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples."""

    @abstractmethod
    def __getitem__(self, idx: int):
        """Return one training sample."""

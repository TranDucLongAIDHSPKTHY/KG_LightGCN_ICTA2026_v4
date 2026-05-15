"""
datasets/dataloader.py
─────────────────────────────────────────────────────────────────────────────
Factory functions for creating DataLoader objects for training and evaluation.
Ensures consistent batch sizes and seeding across all models (fairness).

num_workers > 0 hoạt động đúng trên mọi platform (Windows / Linux / macOS)
vì CFDataset đã implement __getstate__ / __setstate__ để chuyển sparse tensor
thành numpy arrays trước khi pickle sang worker process.
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader

from datasets.cf_dataset import CFDataset
from datasets.kg_dataset import KGDataset


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker for reproducibility."""
    import numpy as np
    np.random.seed(torch.initial_seed() % 2**32)


def get_cf_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 2048,
    neg_samples: int = 1,
    seed: int = 42,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader wrapping CFDataset.

    Args:
        data_dir:    Processed dataset directory.
        split:       'train' | 'val' | 'test'.
        batch_size:  Batch size (fairness: 2048).
        neg_samples: Negatives per positive.
        seed:        Dataset RNG seed.
        num_workers: DataLoader workers (0 = main process).
                     Giá trị > 0 hoạt động trên cả Windows nhờ
                     __getstate__/__setstate__ trong CFDataset.
        shuffle:     Shuffle training data each epoch.

    Returns:
        PyTorch DataLoader.
    """
    dataset = CFDataset(
        data_dir=data_dir,
        split=split,
        neg_samples=neg_samples,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and split == "train"),
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available() and num_workers > 0),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),  # giữ worker alive giữa epochs
        drop_last=False,
    )
    return loader


def get_kg_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 2048,
    neg_samples: int = 1,
    kg_type: str = "full",
    seed: int = 42,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader wrapping KGDataset.

    Args:
        data_dir:    Processed dataset directory.
        split:       'train' | 'val' | 'test'.
        batch_size:  Batch size.
        neg_samples: BPR negatives per positive.
        kg_type:     'full' | 'category' | 'brand' | 'none'.
        seed:        Dataset RNG seed.
        num_workers: DataLoader workers.
        shuffle:     Shuffle training data.

    Returns:
        PyTorch DataLoader wrapping KGDataset.
    """
    dataset = KGDataset(
        data_dir=data_dir,
        split=split,
        neg_samples=neg_samples,
        kg_type=kg_type,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and split == "train"),
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available() and num_workers > 0),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),  # giữ worker alive giữa epochs
        drop_last=False,
    )
    return loader


def build_eval_loader(
    dataset: CFDataset,
    batch_size: int = 512,
) -> DataLoader:
    """
    Wrap a dataset in a non-shuffling DataLoader for evaluation.
    Used by the Evaluator to batch users during full-ranking.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
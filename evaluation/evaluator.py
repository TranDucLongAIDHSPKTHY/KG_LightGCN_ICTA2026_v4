"""
evaluation/evaluator.py
─────────────────────────────────────────────────────────────────────────────
Unified Evaluator: wraps full_ranking_eval and provides a clean interface
used by all trainers. Ensures all models are evaluated identically (fairness).
"""

import os
from typing import Dict, List, Optional

import torch

from evaluation.full_ranking import full_ranking_eval
from utils.logger import get_logger

logger = get_logger("evaluator")


class Evaluator:
    """
    Shared evaluator used by all models and trainers.

    All evaluation calls go through this class to guarantee:
      - Same full-ranking protocol
      - Same top-K list
      - Same train-mask logic
    """

    def __init__(
        self,
        train_user2items: Dict[int, List[int]],
        val_user2items: Dict[int, List[int]],
        test_user2items: Dict[int, List[int]],
        n_items: int,
        device: torch.device,
        batch_size: int = 512,
        top_k_list: List[int] = [10, 20],
    ) -> None:
        """
        Args:
            train_user2items: Training interactions (used for masking).
            val_user2items:   Validation ground truth.
            test_user2items:  Test ground truth.
            n_items:          Total items.
            device:           Evaluation device.
            batch_size:       Users per scoring batch.
            top_k_list:       K values (e.g. [10, 20]).
        """
        self.train_user2items = train_user2items
        self.val_user2items = val_user2items
        self.test_user2items = test_user2items
        self.n_items = n_items
        self.device = device
        self.batch_size = batch_size
        self.top_k_list = top_k_list

    def evaluate(
        self,
        model,
        split: str = "val",
    ) -> Dict[str, float]:
        """
        Run full-ranking evaluation on the given split.

        Args:
            model: Trained model with get_embeddings().
            split: 'val' | 'test'.

        Returns:
            Dict of metric_name → value.
        """
        assert split in ("val", "test"), f"Unknown split: {split}"
        eval_user2items = (
            self.val_user2items if split == "val" else self.test_user2items
        )

        if not eval_user2items:
            logger.warning(f"No evaluation data for split '{split}'.")
            return {}

        metrics = full_ranking_eval(
            model=model,
            train_user2items=self.train_user2items,
            eval_user2items=eval_user2items,
            n_items=self.n_items,
            device=self.device,
            batch_size=self.batch_size,
            top_k_list=self.top_k_list,
        )
        return metrics

    def log_metrics(
        self,
        metrics: Dict[str, float],
        split: str,
        epoch: Optional[int] = None,
    ) -> None:
        prefix = f"[{split.upper()}]" + (f" epoch={epoch}" if epoch is not None else "")
        msg = prefix + "  " + "  ".join(f"{k}={v:.6f}" for k, v in sorted(metrics.items()))
        logger.info(msg)

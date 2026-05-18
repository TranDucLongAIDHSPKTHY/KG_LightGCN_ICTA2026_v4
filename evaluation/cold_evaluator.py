"""
evaluation/cold_evaluator.py
─────────────────────────────────────────────────────────────────────────────
Cold-start evaluator: restricts evaluation to cold items only.
Metrics: HR@10_cold, NDCG@10_cold, Recall@20_cold
"""

import os
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from evaluation.metrics import compute_all_metrics
from evaluation.full_ranking import full_ranking_eval
from utils.logger import get_logger

logger = get_logger("cold_evaluator")


def load_cold_items(cold_dir: str) -> Set[int]:
    """Load cold item IDs from cold_items.txt."""
    path = os.path.join(cold_dir, "cold_items.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"cold_items.txt not found in {cold_dir}")
    with open(path, "r") as f:
        return {int(line.strip()) for line in f if line.strip()}


@torch.no_grad()
def cold_start_eval(
    model,
    train_user2items: Dict[int, List[int]],
    test_user2items: Dict[int, List[int]],
    cold_items: Set[int],
    n_items: int,
    device: torch.device,
    batch_size: int = 512,
) -> Dict[str, float]:
    """
    Cold-start evaluation: only score cold items, evaluate only on cold test pairs.

    Args:
        model:              Trained model.
        train_user2items:   Training interactions (for masking).
        test_user2items:    Test interactions (full — filter cold pairs here).
        cold_items:         Set of cold item IDs.
        n_items:            Total items.
        device:             Torch device.
        batch_size:         User batch size.

    Returns:
        Dict: hr@10_cold, ndcg@10_cold, recall@20_cold
    """
    model.eval()

    # Filter to only users who have cold items in their test set
    cold_test: Dict[int, List[int]] = {}
    for uid, items in test_user2items.items():
        cold_gt = [i for i in items if i in cold_items]
        if cold_gt:
            cold_test[uid] = cold_gt

    if not cold_test:
        logger.warning("No cold test pairs found. Check cold_items.txt.")
        return {}

    logger.info(
        f"Cold eval: {len(cold_test):,} users, {len(cold_items):,} cold items"
    )

    user_emb, item_emb = model.get_embeddings()
    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)

    cold_item_list = sorted(cold_items)
    cold_item_tensor = torch.tensor(cold_item_list, dtype=torch.long, device=device)

    eval_users = sorted(cold_test.keys())
    all_ranked: List[np.ndarray] = []
    all_gt: List[List[int]] = []

    top_k_list = [10, 20]
    max_k = max(top_k_list)

    for start in range(0, len(eval_users), batch_size):
        batch_users = eval_users[start: start + batch_size]
        u_emb = user_emb[batch_users]  # [B, D]

        # Score ONLY cold items: [B, |cold_items|]
        cold_item_emb = item_emb[cold_item_tensor]  # [|cold|, D]
        scores = torch.matmul(u_emb, cold_item_emb.T)  # [B, |cold|]

        # Mask train cold items (user already saw them in train)
        for local_i, uid in enumerate(batch_users):
            train_items = set(train_user2items.get(uid, []))
            for j, cold_iid in enumerate(cold_item_list):
                if cold_iid in train_items:
                    scores[local_i, j] = float("-inf")

        k_eff = min(max_k, len(cold_item_list))
        _, ranked_local = torch.topk(scores, k=k_eff, dim=-1)
        # Convert local cold indices back to global item IDs
        # ranked_global = cold_item_tensor[ranked_local.cpu()].numpy()  # [B, k]
        ranked_global = cold_item_tensor.cpu()[ranked_local.cpu()].numpy()

        for local_i, uid in enumerate(batch_users):
            all_ranked.append(ranked_global[local_i])
            all_gt.append(cold_test[uid])

    ranked_matrix = np.vstack(all_ranked)  # [N_cold_users, max_k]
    metrics_raw = compute_all_metrics(ranked_matrix, all_gt, top_k_list)

    # Rename to indicate cold evaluation
    cold_metrics = {f"{k}_cold": v for k, v in metrics_raw.items()}
    return cold_metrics


class ColdEvaluator:
    """
    Wrapper class for cold-start evaluation.

    Usage (from directory):
        evaluator = ColdEvaluator(cold_dir, train_d, test_d, n_items, device)

    Usage (from pre-built cold_items set):
        evaluator = ColdEvaluator(
            cold_dir_or_items=cold_items_set,
            train_user2items=train_d,
            test_user2items=test_d,
            n_items=n_items,
            device=device,
            top_k_list=[10, 20],
        )
    """

    def __init__(
        self,
        cold_dir_or_items,          # str path OR Set[int] of cold item IDs
        train_user2items: Dict[int, List[int]],
        test_user2items: Dict[int, List[int]],
        n_items: int,
        device: torch.device,
        batch_size: int = 512,
        top_k_list: Optional[List[int]] = None,  # optional, kept for API compat
    ) -> None:
        if isinstance(cold_dir_or_items, (set, frozenset)):
            self.cold_items: Set[int] = cold_dir_or_items
        else:
            self.cold_items = load_cold_items(str(cold_dir_or_items))
        self.train_user2items = train_user2items
        self.test_user2items = test_user2items
        self.n_items = n_items
        self.device = device
        self.batch_size = batch_size
        logger.info(f"ColdEvaluator initialised: {len(self.cold_items):,} cold items")

    def evaluate(self, model) -> Dict[str, float]:
        return cold_start_eval(
            model=model,
            train_user2items=self.train_user2items,
            test_user2items=self.test_user2items,
            cold_items=self.cold_items,
            n_items=self.n_items,
            device=self.device,
            batch_size=self.batch_size,
        )

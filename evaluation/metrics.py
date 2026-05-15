"""
evaluation/metrics.py
─────────────────────────────────────────────────────────────────────────────
Vectorised implementation of recommendation metrics:
  - Recall@K
  - NDCG@K
  - HR@K (Hit Rate)

All functions operate on pre-ranked item lists.
"""

import numpy as np
from typing import Dict, List


def recall_at_k(
    ranked_items: np.ndarray,
    ground_truth: List[int],
    k: int,
) -> float:
    """
    Recall@K for a single user.

    Args:
        ranked_items: 1-D array of item IDs sorted by descending score.
        ground_truth: List of relevant item IDs.
        k:            Cut-off.

    Returns:
        Recall@K ∈ [0, 1].
    """
    if not ground_truth:
        return 0.0
    top_k = set(ranked_items[:k].tolist())
    gt = set(ground_truth)
    # FIX: standard Recall@K = |hits| / |GT|, not / min(|GT|, k)
    # Dividing by min(len(gt), k) computes "capped recall" which inflates
    # scores for users with few ground-truth items and is non-standard.
    return len(top_k & gt) / len(gt)


def ndcg_at_k(
    ranked_items: np.ndarray,
    ground_truth: List[int],
    k: int,
) -> float:
    """
    NDCG@K for a single user (binary relevance).

    Args:
        ranked_items: 1-D array of item IDs sorted by descending score.
        ground_truth: List of relevant item IDs.
        k:            Cut-off.

    Returns:
        NDCG@K ∈ [0, 1].
    """
    if not ground_truth:
        return 0.0
    gt = set(ground_truth)
    top_k = ranked_items[:k].tolist()

    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item in enumerate(top_k)
        if item in gt
    )
    # Ideal DCG: all relevant items at top positions
    ideal_k = min(len(gt), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_k))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(
    ranked_items: np.ndarray,
    ground_truth: List[int],
    k: int,
) -> float:
    """
    HR@K (Hit Rate) for a single user.
    1 if at least one relevant item is in top-K, else 0.
    """
    if not ground_truth:
        return 0.0
    top_k = set(ranked_items[:k].tolist())
    gt = set(ground_truth)
    return 1.0 if top_k & gt else 0.0


# ── Batch (vectorised) versions ───────────────────────────────────────────────

def batch_recall_at_k(
    ranked_matrix: np.ndarray,
    ground_truths: List[List[int]],
    k: int,
) -> np.ndarray:
    """
    Recall@K for a batch of users.

    Args:
        ranked_matrix: [B, n_items] item IDs sorted by score per row.
        ground_truths: List of B ground-truth lists.
        k:             Cut-off.

    Returns:
        recalls: [B] float array.
    """
    recalls = np.zeros(len(ground_truths), dtype=np.float32)
    for i, gt in enumerate(ground_truths):
        if not gt:
            continue
        gt_set = set(gt)
        row = ranked_matrix[i]
        actual_k = min(k, len(row))
        hits = sum(1 for item in row[:actual_k] if item in gt_set)
        # FIX: denominator is |GT|, not min(|GT|, k)
        recalls[i] = hits / len(gt_set)
    return recalls


def batch_ndcg_at_k(
    ranked_matrix: np.ndarray,
    ground_truths: List[List[int]],
    k: int,
) -> np.ndarray:
    """
    NDCG@K for a batch of users.

    Returns:
        ndcgs: [B] float array.
    """
    ndcgs = np.zeros(len(ground_truths), dtype=np.float32)
    log2 = np.log2(np.arange(2, k + 2))  # precompute log2(2)…log2(k+1) — length k

    for i, gt in enumerate(ground_truths):
        if not gt:
            continue
        gt_set = set(gt)
        # ranked_matrix may have fewer than k items if n_items < k — slice safely
        row = ranked_matrix[i]
        actual_k = min(k, len(row))
        hits = np.array([1.0 if item in gt_set else 0.0 for item in row[:actual_k]])
        log2_k = log2[:actual_k]  # match length to actual items returned
        dcg = (hits / log2_k).sum()
        ideal_k = min(len(gt_set), actual_k)
        idcg = (1.0 / log2[:ideal_k]).sum()
        ndcgs[i] = dcg / idcg if idcg > 0 else 0.0
    return ndcgs


def batch_hr_at_k(
    ranked_matrix: np.ndarray,
    ground_truths: List[List[int]],
    k: int,
) -> np.ndarray:
    """
    HR@K for a batch of users.

    Returns:
        hrs: [B] float array.
    """
    hrs = np.zeros(len(ground_truths), dtype=np.float32)
    for i, gt in enumerate(ground_truths):
        if not gt:
            continue
        gt_set = set(gt)
        row = ranked_matrix[i]
        actual_k = min(k, len(row))
        hrs[i] = 1.0 if any(item in gt_set for item in row[:actual_k]) else 0.0
    return hrs


def compute_all_metrics(
    ranked_matrix: np.ndarray,
    ground_truths: List[List[int]],
    top_k_list: List[int] = [10, 20],
) -> Dict[str, float]:
    """
    Compute Recall@K, NDCG@K, HR@K for multiple cut-offs.

    Args:
        ranked_matrix: [B, n_items] sorted item IDs.
        ground_truths: B ground-truth lists.
        top_k_list:    List of K values (e.g. [10, 20]).

    Returns:
        Dict: e.g. {'recall@20': 0.12, 'ndcg@20': 0.09, 'hr@10': 0.25, …}
    """
    results: Dict[str, float] = {}
    for k in top_k_list:
        recalls = batch_recall_at_k(ranked_matrix, ground_truths, k)
        ndcgs = batch_ndcg_at_k(ranked_matrix, ground_truths, k)
        hrs = batch_hr_at_k(ranked_matrix, ground_truths, k)
        results[f"recall@{k}"] = float(recalls.mean())
        results[f"ndcg@{k}"] = float(ndcgs.mean())
        results[f"hr@{k}"] = float(hrs.mean())
    return results
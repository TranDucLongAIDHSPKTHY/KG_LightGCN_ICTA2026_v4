"""
evaluation/metrics.py
─────────────────────────────────────────────────────────────────────────────
Vectorised implementation of recommendation metrics:
  - Recall@K
  - NDCG@K
  - HR@K (Hit Rate)

All functions operate on pre-ranked item lists.

─────────────────────────────────────────────────────────────────────────────
RECALL@K — PAPER FORMULA (LightGCN / KGCL / SimGCL)
─────────────────────────────────────────────────────────────────────────────

        Recall@K = |hits ∩ GT| / min(|GT|, K)

Mirrors gusye1234/LightGCN-PyTorch utils.py — RecallPrecision_ATk:
    recall = right_items / min(len(ground_true), K)

On Amazon-Book most test users have exactly 1 item →
min(1, 20) = 1 → Recall@20 ≈ HR@20.

NDCG@K — also matches LightGCN paper repo (NDCGatK_r in utils.py):
    DCG  = Σ rel_i / log2(i + 2)          (i = 0-indexed position in top-K)
    IDCG = Σ_{i=0}^{min(|GT|,K)-1} 1 / log2(i + 2)
"""

import numpy as np
from typing import Dict, List


# ── Single-user functions ─────────────────────────────────────────────────────

def recall_at_k(
    ranked_items: np.ndarray,
    ground_truth: List[int],
    k: int,
) -> float:
    """
    Recall@K for a single user (LightGCN paper formula).

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
    hits = len(top_k & gt)
    denom = min(len(gt), k)
    return hits / denom if denom > 0 else 0.0


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
    Returns 1 if at least one relevant item is in top-K, else 0.
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
    Recall@K for a batch of users (LightGCN paper formula).

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
        denom = min(len(gt_set), k)
        recalls[i] = hits / denom if denom > 0 else 0.0
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
    # Precompute log2(2) … log2(k+1), length k
    log2_table = np.log2(np.arange(2, k + 2))

    for i, gt in enumerate(ground_truths):
        if not gt:
            continue
        gt_set = set(gt)
        row = ranked_matrix[i]
        actual_k = min(k, len(row))

        hits = np.array(
            [1.0 if item in gt_set else 0.0 for item in row[:actual_k]],
            dtype=np.float32,
        )
        dcg = (hits / log2_table[:actual_k]).sum()

        ideal_k = min(len(gt_set), actual_k)
        idcg = (1.0 / log2_table[:ideal_k]).sum()

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


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_all_metrics(
    ranked_matrix: np.ndarray,
    ground_truths: List[List[int]],
    top_k_list: List[int] = [10, 20],
) -> Dict[str, float]:
    """
    Compute Recall@K, NDCG@K, HR@K for multiple cut-off values.

    All metrics use the LightGCN paper formula — results are directly
    comparable with LightGCN / KGCL / SimGCL reported numbers.

    Args:
        ranked_matrix: [B, n_items] sorted item IDs.
        ground_truths: B ground-truth lists.
        top_k_list:    K values to evaluate (e.g. [10, 20]).

    Returns:
        Dict: e.g. {'recall@20': 0.12, 'ndcg@20': 0.09, 'hr@10': 0.25, …}

    Example:
        metrics = compute_all_metrics(ranked, gt, [10, 20])
    """
    results: Dict[str, float] = {}
    for k in top_k_list:
        recalls = batch_recall_at_k(ranked_matrix, ground_truths, k)
        ndcgs   = batch_ndcg_at_k(ranked_matrix, ground_truths, k)
        hrs     = batch_hr_at_k(ranked_matrix, ground_truths, k)
        results[f"recall@{k}"] = float(recalls.mean())
        results[f"ndcg@{k}"]   = float(ndcgs.mean())
        results[f"hr@{k}"]     = float(hrs.mean())
    return results
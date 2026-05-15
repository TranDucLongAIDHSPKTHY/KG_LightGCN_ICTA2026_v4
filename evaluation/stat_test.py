"""
evaluation/stat_test.py
─────────────────────────────────────────────────────────────────────────────
Statistical significance tests for comparing recommender models.

Implements:
  - Paired t-test (p < 0.05)
  - Cohen's d (effect size)

Comparisons:
  - KG-LightGCN vs LightGCN
  - KG-LightGCN vs KGCL
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two paired arrays.

    d = mean(a - b) / std(a - b)

    Args:
        a: [N] scores for model A.
        b: [N] scores for model B.

    Returns:
        Cohen's d (positive = A > B).
    """
    diff = a - b
    std = diff.std(ddof=1)
    if std == 0:
        return 0.0
    return diff.mean() / std


def paired_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Paired t-test between two sets of per-user scores.

    Args:
        scores_a: [N] per-user metric values for model A.
        scores_b: [N] per-user metric values for model B.
        alpha:    Significance threshold.

    Returns:
        Dict with t_stat, p_value, cohen_d, significant (bool), mean_diff.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Score arrays must have equal length: {len(scores_a)} vs {len(scores_b)}"
        )

    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    d = cohens_d(scores_a, scores_b)
    mean_diff = float(np.mean(scores_a) - np.mean(scores_b))

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohen_d": float(d),
        "significant": bool(p_value < alpha),
        "mean_diff": mean_diff,
        "mean_a": float(np.mean(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "alpha": alpha,
    }


def compare_models(
    results_a: Dict[str, List[float]],
    results_b: Dict[str, List[float]],
    model_a_name: str = "KG-LightGCN",
    model_b_name: str = "LightGCN",
    metrics: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> Dict[str, Dict]:
    """
    Run paired t-tests across multiple metrics between two models.

    Args:
        results_a:    {metric: [seed_run_scores]} for model A.
        results_b:    {metric: [seed_run_scores]} for model B.
        model_a_name: Display name for model A.
        model_b_name: Display name for model B.
        metrics:      List of metric keys to test. If None, use all in results_a.
        alpha:        Significance level.

    Returns:
        {metric: {t_stat, p_value, cohen_d, significant, …}}
    """
    if metrics is None:
        metrics = list(results_a.keys())

    comparison: Dict[str, Dict] = {}
    for metric in metrics:
        if metric not in results_a or metric not in results_b:
            continue
        a = np.array(results_a[metric], dtype=np.float64)
        b = np.array(results_b[metric], dtype=np.float64)
        result = paired_ttest(a, b, alpha=alpha)
        result["model_a"] = model_a_name
        result["model_b"] = model_b_name
        result["metric"] = metric
        comparison[metric] = result

    return comparison


def print_significance_report(
    comparison: Dict[str, Dict],
    model_a_name: str = "KG-LightGCN",
    model_b_name: str = "Baseline",
) -> str:
    """
    Format a human-readable significance report.

    Returns:
        Report string (also prints to stdout).
    """
    lines = [
        "=" * 70,
        f"SIGNIFICANCE TEST: {model_a_name} vs {model_b_name}",
        "=" * 70,
        f"{'Metric':<20} {'Mean A':>10} {'Mean B':>10} {'Diff':>10} "
        f"{'t-stat':>10} {'p-value':>10} {'Cohen d':>10} {'Sig':>5}",
        "-" * 70,
    ]
    for metric, res in sorted(comparison.items()):
        sig = "✓" if res["significant"] else "✗"
        lines.append(
            f"{metric:<20} {res['mean_a']:>10.6f} {res['mean_b']:>10.6f} "
            f"{res['mean_diff']:>+10.6f} {res['t_statistic']:>10.4f} "
            f"{res['p_value']:>10.4f} {res['cohen_d']:>10.4f} {sig:>5}"
        )
    lines.append("=" * 70)
    report = "\n".join(lines)
    print(report)
    return report


def save_significance_results(
    comparison: Dict[str, Dict],
    output_path: str,
) -> None:
    """Save significance test results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

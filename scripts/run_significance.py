"""
scripts/run_significance.py
─────────────────────────────────────────────────────────────────────────────
Load multi-seed results and run paired t-tests.

Comparisons (per pipeline spec):
  - KG-LightGCN vs LightGCN
  - KG-LightGCN vs KGCL

Usage:
  python scripts/run_significance.py --dataset amazon-book
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.stat_test import compare_models, print_significance_report, save_significance_results
from utils.logger import get_logger

logger = get_logger("run_significance")

METRICS = ["recall@20", "ndcg@20", "hr@10", "ndcg@10"]


def load_per_seed_metrics(result_path: str) -> dict:
    """
    Load per-seed test metrics from a results JSON file.

    Expected format (saved by Trainer):
        results/tables/<model>_results.json
        {
          "mean": {...},
          "std":  {...},
          "per_seed": [ {"test_metrics": {...}}, ... ]
        }
    Returns: {metric: [seed1_val, seed2_val, ...]}
    """
    if not os.path.exists(result_path):
        return {}
    with open(result_path) as f:
        data = json.load(f)
    per_seed = data.get("per_seed", [])
    if not per_seed:
        # Fallback: use mean as a single value
        return {k: [v] for k, v in data.get("mean", {}).items()}

    metric_lists = {}
    for seed_result in per_seed:
        for k, v in seed_result.get("test_metrics", {}).items():
            metric_lists.setdefault(k, []).append(v)
    return metric_lists


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="amazon-book")
    parser.add_argument("--result_dir", default="results/tables")
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = args.result_dir

    pairs = [
        ("kg_lightgcn",    "lightgcn"),      # Base vs LightGCN
        ("kg_lightgcn",    "kgcl"),           # Base vs KGCL
        ("kg_lightgcn_cl", "lightgcn"),      # Enhanced vs LightGCN
        ("kg_lightgcn_cl", "kgcl"),           # Enhanced vs KGCL
        ("kg_lightgcn_cl", "kg_lightgcn"),   # Enhanced vs Base (ablation)
    ]

    for model_a, model_b in pairs:
        path_a = os.path.join(result_dir, f"{model_a}_results.json")
        path_b = os.path.join(result_dir, f"{model_b}_results.json")

        res_a = load_per_seed_metrics(path_a)
        res_b = load_per_seed_metrics(path_b)

        if not res_a:
            logger.warning(f"No results found for {model_a}: {path_a}")
            continue
        if not res_b:
            logger.warning(f"No results found for {model_b}: {path_b}")
            continue

        available = [m for m in METRICS if m in res_a and m in res_b]
        if not available:
            logger.warning(f"No shared metrics between {model_a} and {model_b}.")
            continue

        comparison = compare_models(
            results_a=res_a,
            results_b=res_b,
            model_a_name=model_a,
            model_b_name=model_b,
            metrics=available,
            alpha=args.alpha,
        )

        report = print_significance_report(comparison, model_a, model_b)

        out_path = os.path.join(
            result_dir, f"significance_{model_a}_vs_{model_b}.json"
        )
        save_significance_results(comparison, out_path)
        logger.info(f"Saved significance results to {out_path}")


if __name__ == "__main__":
    main()

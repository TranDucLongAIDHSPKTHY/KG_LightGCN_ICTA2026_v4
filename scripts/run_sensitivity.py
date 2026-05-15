"""
scripts/run_sensitivity.py
─────────────────────────────────────────────────────────────────────────────
Sensitivity analysis for KG-LightGCN:
  - K (n_layers) ∈ {1, 2, 3, 4}
  - d (embedding_dim) ∈ {32, 64, 128, 256}
  - λ (weight_decay) ∈ {0.01, 0.1, 1.0}

Uses only seed=42 for speed (full multi-seed too expensive).

Usage:
  python scripts/run_sensitivity.py --dataset amazon-book --param n_layers
  python scripts/run_sensitivity.py --dataset amazon-book --param all
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import train_model
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("sensitivity")

SENSITIVITY_GRID = {
    "n_layers": [1, 2, 3, 4],
    "embedding_dim": [32, 64, 128, 256],
    "weight_decay": [0.01, 0.1, 1.0],
}


def run_sensitivity(param: str, values, dataset: str, result_dir: str):
    """Run sensitivity sweep for one parameter."""
    logger.info(f"\n{'='*60}\nSensitivity: {param} ∈ {values}\n{'='*60}")
    sweep_results = {}

    for val in values:
        logger.info(f"  {param} = {val}")
        overrides = {
            "dataset.name": dataset,
            f"model.{param}" if param != "weight_decay" else f"train.{param}": val,
        }
        # embedding_dim override needs to disable fairness lock for sensitivity only
        if param == "embedding_dim":
            overrides["model.embedding_dim"] = val

        cfg = load_config(
            base_path="configs/base.yaml",
            model_config_path="configs/model/kg_lightgcn.yaml",
            overrides=overrides,
        )
        results = train_model(
            model_name="kg_lightgcn",
            cfg=cfg,
            seeds=[42],  # single seed for sensitivity
        )
        sweep_results[str(val)] = results.get("mean", {})

    # Save
    out_path = os.path.join(result_dir, f"sensitivity_{param}_{dataset}.json")
    os.makedirs(result_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    logger.info(f"Saved sensitivity results: {out_path}")

    # Print
    print(f"\n{'='*60}")
    print(f"Sensitivity: {param}")
    print(f"{'='*60}")
    metric = "recall@20"
    for val, metrics in sweep_results.items():
        print(f"  {param}={val:>6}  {metric}={metrics.get(metric, 'N/A'):.6f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="amazon-book")
    parser.add_argument(
        "--param",
        choices=list(SENSITIVITY_GRID.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--result_dir", default="results/tables")
    return parser.parse_args()


def main():
    args = parse_args()
    params = list(SENSITIVITY_GRID.keys()) if args.param == "all" else [args.param]

    for param in params:
        run_sensitivity(
            param=param,
            values=SENSITIVITY_GRID[param],
            dataset=args.dataset,
            result_dir=args.result_dir,
        )


if __name__ == "__main__":
    main()

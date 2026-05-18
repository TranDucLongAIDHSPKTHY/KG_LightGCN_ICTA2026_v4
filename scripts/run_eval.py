"""
scripts/run_eval.py
─────────────────────────────────────────────────────────────────────────────
Standalone evaluation: load a saved checkpoint and run full-ranking + cold eval.

Usage:
  python scripts/run_eval.py \
      --model lightgcn \
      --dataset amazon-book \
      --checkpoint results/checkpoints/lightgcn_seed42_best.pt \
      --split test

  python scripts/run_eval.py \
      --model kg_lightgcn \
      --dataset amazon-book \
      --checkpoint results/checkpoints/kg_lightgcn_seed42_best.pt \
      --cold_dir data/processed/amazon-book/cold_20
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    get_device,
    build_cf_dataset,
    build_kg_dataset,
    build_evaluator,
    build_lightgcn,
    build_simgcl,
    build_kgat,
    build_kgcl,
    build_kg_lightgcn,
    MODEL_BUILDERS,
    KG_MODELS,
)
from evaluation.cold_evaluator import ColdEvaluator
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("run_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="amazon-book")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--cold_dir", default=None, help="Path to cold split dir.")
    parser.add_argument("--base_config", default="configs/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_cfg_path = f"configs/model/{args.model}.yaml"
    cfg = load_config(
        base_path=args.base_config,
        model_config_path=model_cfg_path if os.path.exists(model_cfg_path) else None,
        overrides={"dataset.name": args.dataset},
    )

    device = get_device(cfg)
    data_dir = os.path.join(
        cfg.get("dataset", {}).get("data_dir", "/data/phuongtran/processed"),
        args.dataset,
    )

    _, model_type = MODEL_BUILDERS[args.model]

    if model_type == "cf":
        dataset = build_cf_dataset(data_dir, cfg)
    else:
        dataset = build_kg_dataset(data_dir, cfg)

    evaluator = build_evaluator(dataset, cfg, device)

    # Build model
    builder_fn, _ = MODEL_BUILDERS[args.model]
    model = builder_fn(dataset, cfg, device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Full ranking eval
    metrics = evaluator.evaluate(model, split=args.split)
    evaluator.log_metrics(metrics, split=args.split)

    # Cold eval
    if args.cold_dir and os.path.isdir(args.cold_dir):
        train_d = dataset.read_interaction_file(os.path.join(data_dir, "train.txt"))
        test_d = dataset.read_interaction_file(os.path.join(data_dir, "test.txt"))
        cold_evaluator = ColdEvaluator(
            cold_dir=args.cold_dir,
            train_user2items=train_d,
            test_user2items=test_d,
            n_items=dataset.n_items,
            device=device,
        )
        cold_metrics = cold_evaluator.evaluate(model)
        logger.info(f"Cold metrics: {cold_metrics}")
        metrics.update(cold_metrics)

    out_path = f"results/tables/{args.model}_{args.split}_eval.json"
    os.makedirs("results/tables", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

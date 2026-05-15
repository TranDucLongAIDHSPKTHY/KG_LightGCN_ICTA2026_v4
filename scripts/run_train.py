"""
scripts/run_train.py
─────────────────────────────────────────────────────────────────────────────
Convenience wrapper around main.py for training a single model.
Mainly useful for batch scripts / SLURM jobs.

Usage:
  python scripts/run_train.py --model lightgcn --dataset amazon-book --seed 42
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import train_model
from utils.config import load_config
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger("run_train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="amazon-book")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cold_split", default=None)
    parser.add_argument("--kg_type", default=None)
    parser.add_argument("--base_config", default="configs/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model_config_path = f"configs/model/{args.model}.yaml"
    overrides = {"dataset.name": args.dataset}
    if args.kg_type:
        overrides["dataset.kg_type"] = args.kg_type

    cfg = load_config(
        base_path=args.base_config,
        model_config_path=model_config_path if os.path.exists(model_config_path) else None,
        overrides=overrides,
    )

    logger.info(f"Training {args.model} on {args.dataset} | seed={args.seed}")
    result = train_model(
        model_name=args.model,
        cfg=cfg,
        seeds=[args.seed],
        cold_split=args.cold_split,
    )
    logger.info(f"Result: {result.get('mean')}")


if __name__ == "__main__":
    main()

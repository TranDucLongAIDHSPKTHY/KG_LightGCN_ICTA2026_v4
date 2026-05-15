"""
main.py
─────────────────────────────────────────────────────────────────────────────
Unified entry point for the KG-LightGCN project.

Usage examples:
  # Train LightGCN on Amazon-Book
  python main.py --model lightgcn --dataset amazon-book

  # Train KG-LightGCN with cold-20 evaluation
  python main.py --model kg_lightgcn --dataset amazon-book --cold_split cold_20

  # Train all models, all seeds
  python main.py --model all --dataset amazon-book --seeds 42 0 1 2 3

  # KG ablation: category only
  python main.py --model kg_lightgcn --dataset amazon-book --kg_type category

  # Override hyperparameters
  python main.py --model lightgcn --dataset yelp2018 --override train.n_layers=4
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

from datasets.cf_dataset import CFDataset
from datasets.kg_dataset import KGDataset
from datasets.dataloader import get_cf_dataloader, get_kg_dataloader
from evaluation.evaluator import Evaluator
from evaluation.cold_evaluator import ColdEvaluator
from models.lightgcn import LightGCN
from models.simgcl import SimGCL
from models.kgat import KGAT
from models.kgcl import KGCL
from models.kg_lightgcn import KGLightGCN, KGLightGCNCL
from trainers.trainer import Trainer, run_multi_seed
from trainers.kg_trainer import KGTrainer, run_kg_multi_seed
from utils.config import load_config, save_config
from utils.logger import get_logger
from utils.seed import set_seed, get_seeds

logger = get_logger("main")

CF_MODELS = {"lightgcn", "simgcl"}
KG_MODELS = {"kgat", "kgcl", "kg_lightgcn", "kg_lightgcn_cl"}
ALL_MODELS = CF_MODELS | KG_MODELS


def get_device(cfg: dict) -> torch.device:
    """Resolve device from config (auto | cpu | cuda)."""
    dev = cfg.get("train", {}).get("device", "auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def get_data_dir(cfg: dict, cold_split: Optional[str] = None) -> str:
    """
    Build data directory path.

    Thứ tự ưu tiên:
      1. DATA_ROOT env var (set trong .env hoặc shell)  → DATA_ROOT/<dataset_name>
      2. dataset.data_dir trong config YAML             → data_dir/<dataset_name>
      3. Fallback mặc định                              → data/processed/<dataset_name>

    Không hard-code path — dùng .env để chỉ định nơi lưu dataset.
    """
    # data_dir đã được resolve bởi config.py (DATA_ROOT → dataset.data_dir)
    data_dir = cfg.get("dataset", {}).get("data_dir", "data/processed")
    dataset_name = cfg.get("dataset", {}).get("name", "amazon-book")
    base = os.path.join(data_dir, dataset_name)
    if cold_split:
        return os.path.join(base, cold_split)
    return base


def build_cf_dataset(data_dir: str, cfg: dict) -> CFDataset:
    """Create CFDataset from config (all splits loaded at once)."""
    seed = cfg.get("train", {}).get("seed", 42)
    return CFDataset(data_dir=data_dir, split="train", seed=seed)


def build_kg_dataset(data_dir: str, cfg: dict) -> KGDataset:
    """Create KGDataset from config."""
    kg_type = cfg.get("dataset", {}).get("kg_type", "full")
    seed = cfg.get("train", {}).get("seed", 42)
    return KGDataset(data_dir=data_dir, split="train", kg_type=kg_type, seed=seed)


def build_evaluator(dataset: CFDataset, cfg: dict, device: torch.device) -> Evaluator:
    """Build shared Evaluator from a loaded dataset."""
    train_d = dataset.read_interaction_file(
        os.path.join(dataset.data_dir, "train.txt")
    )
    val_d = dataset.read_interaction_file(
        os.path.join(dataset.data_dir, "val.txt")
    )
    test_d = dataset.read_interaction_file(
        os.path.join(dataset.data_dir, "test.txt")
    )
    eval_cfg = cfg.get("eval", {})
    return Evaluator(
        train_user2items=train_d,
        val_user2items=val_d,
        test_user2items=test_d,
        n_items=dataset.n_items,
        device=device,
        batch_size=eval_cfg.get("batch_size", 2048),
        top_k_list=eval_cfg.get("top_k", [10, 20]),
    )


# ── Model builders ────────────────────────────────────────────────────────────

def build_lightgcn(dataset: CFDataset, cfg: dict, device: torch.device) -> LightGCN:
    model_cfg = cfg.get("model", {})
    return LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_dim=model_cfg.get("embedding_dim", 64),
        n_layers=model_cfg.get("n_layers", 3),
        norm_adj=dataset.norm_adj_mat,
        device=device,
    ).to(device)


def build_simgcl(dataset: CFDataset, cfg: dict, device: torch.device) -> SimGCL:
    model_cfg = cfg.get("model", {})
    cl_cfg = cfg.get("contrastive", {})
    return SimGCL(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_dim=model_cfg.get("embedding_dim", 64),
        n_layers=model_cfg.get("n_layers", 3),
        eps=model_cfg.get("eps", 0.1),
        temperature=cl_cfg.get("temperature", 0.2),
        lambda_cl=cl_cfg.get("lambda_cl", 0.5),
        norm_adj=dataset.norm_adj_mat,
        device=device,
    ).to(device)


def build_kgat(dataset: KGDataset, cfg: dict, device: torch.device) -> KGAT:
    model_cfg = cfg.get("model", {})
    model = KGAT(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        n_entities=dataset.n_entities,
        n_relations=dataset.n_relations,
        embedding_dim=model_cfg.get("embedding_dim", 64),
        relation_dim=model_cfg.get("relation_dim", 64),
        n_layers=model_cfg.get("n_layers", 3),
        agg_type=model_cfg.get("agg_type", "bi-interaction"),
        norm_adj=dataset.norm_adj_mat,
        device=device,
    ).to(device)
    # Set KG adjacency list for attention aggregation
    model.set_kg_adj(dataset.build_kg_adj_list())
    return model


def build_kgcl(dataset: KGDataset, cfg: dict, device: torch.device) -> KGCL:
    model_cfg = cfg.get("model", {})
    cl_cfg = cfg.get("contrastive", {})
    model = KGCL(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        n_entities=dataset.n_entities,
        n_relations=dataset.n_relations,
        embedding_dim=model_cfg.get("embedding_dim", 64),
        n_layers=model_cfg.get("n_layers", 3),
        kg_n_layers=model_cfg.get("kg_n_layers", 2),
        temp=cl_cfg.get("temperature", 0.2),
        lambda_kg=cl_cfg.get("lambda_cl", 0.5),
        norm_adj=dataset.norm_adj_mat,
        kg_triples=dataset.kg_triples,
        device=device,
    ).to(device)
    # Build and set KG entity adjacency
    kg_sparse = dataset.build_kg_sparse_adj()
    from datasets.cf_dataset import CFDataset as _CFD
    kg_norm = _CFD._sparse_mx_to_torch(kg_sparse).to(device)
    model.set_kg_norm_adj(kg_norm)
    return model


def build_kg_lightgcn(dataset: KGDataset, cfg: dict, device: torch.device) -> KGLightGCN:
    model_cfg = cfg.get("model", {})
    model = KGLightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        n_entities=dataset.n_entities,
        n_relations=dataset.n_relations,
        embedding_dim=model_cfg.get("embedding_dim", 64),
        n_layers=model_cfg.get("n_layers", 3),
        kg_n_layers=model_cfg.get("kg_n_layers", 2),
        kg_type=cfg.get("dataset", {}).get("kg_type", "full"),
        entity_agg=model_cfg.get("entity_agg", "mean"),
        kg_reg=model_cfg.get("kg_reg", 1e-5),
        norm_adj=dataset.norm_adj_mat,
        device=device,
    ).to(device)

    # KG entity adjacency
    kg_sparse = dataset.build_kg_sparse_adj()
    from datasets.cf_dataset import CFDataset as _CFD
    kg_norm = _CFD._sparse_mx_to_torch(kg_sparse).to(device)
    model.set_kg_norm_adj(kg_norm)

    # Item→entity mapping
    if dataset.item2entity:
        item2entity_arr = torch.full((dataset.n_items,), -1, dtype=torch.long)
        for iid, eid in dataset.item2entity.items():
            if 0 <= iid < dataset.n_items:
                item2entity_arr[iid] = eid
        model.set_item_entity_map(item2entity_arr.to(device))

    return model




def build_kg_lightgcn_cl(dataset: KGDataset, cfg: dict, device: torch.device) -> KGLightGCNCL:
    """
    Build KGLightGCNCL (Biến thể 2 - Enhanced/Proposed).
    Giống build_kg_lightgcn nhưng thêm CL hyperparameters.
    """
    model_cfg = cfg.get("model", {})
    cl_cfg = cfg.get("contrastive", {})
    model = KGLightGCNCL(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        n_entities=dataset.n_entities,
        n_relations=dataset.n_relations,
        embedding_dim=model_cfg.get("embedding_dim", 64),
        n_layers=model_cfg.get("n_layers", 3),
        kg_n_layers=model_cfg.get("kg_n_layers", 2),
        kg_type=cfg.get("dataset", {}).get("kg_type", "full"),
        entity_agg=model_cfg.get("entity_agg", "mean"),
        kg_reg=model_cfg.get("kg_reg", 1e-5),
        cl_temp=cl_cfg.get("temperature", 0.2),
        lambda_cl=cl_cfg.get("lambda_cl", 0.5),
        eps=model_cfg.get("eps", 0.1),
        norm_adj=dataset.norm_adj_mat,
        device=device,
    ).to(device)

    # KG entity adjacency
    kg_sparse = dataset.build_kg_sparse_adj()
    from datasets.cf_dataset import CFDataset as _CFD
    kg_norm = _CFD._sparse_mx_to_torch(kg_sparse).to(device)
    model.set_kg_norm_adj(kg_norm)

    # Item -> entity mapping
    if dataset.item2entity:
        item2entity_arr = torch.full((dataset.n_items,), -1, dtype=torch.long)
        for iid, eid in dataset.item2entity.items():
            if 0 <= iid < dataset.n_items:
                item2entity_arr[iid] = eid
        model.set_item_entity_map(item2entity_arr.to(device))

    return model

MODEL_BUILDERS = {
    "lightgcn": (build_lightgcn, "cf"),
    "simgcl": (build_simgcl, "cf"),
    "kgat": (build_kgat, "kg"),
    "kgcl": (build_kgcl, "kg"),
    "kg_lightgcn": (build_kg_lightgcn, "kg"),
    "kg_lightgcn_cl": (build_kg_lightgcn_cl, "kg"),
}


# ── Main training function ────────────────────────────────────────────────────

def train_model(
    model_name: str,
    cfg: dict,
    seeds: List[int],
    cold_split: Optional[str] = None,
) -> Dict:
    """Train a single model with multi-seed support."""
    device = get_device(cfg)
    data_dir = get_data_dir(cfg, cold_split)
    result_dir = cfg.get("logging", {}).get("result_dir", "results/tables")
    checkpoint_dir = cfg.get("logging", {}).get("checkpoint_dir", "results/checkpoints")
    log_dir = cfg.get("logging", {}).get("log_dir", "results/logs")
    os.makedirs(result_dir, exist_ok=True)

    logger.info(f"Model: {model_name} | Dataset: {data_dir} | Device: {device}")

    builder_fn, model_type = MODEL_BUILDERS[model_name]

    if model_type == "cf":
        dataset = build_cf_dataset(data_dir, cfg)
        evaluator = build_evaluator(dataset, cfg, device)

        def model_factory():
            return builder_fn(dataset, cfg, device)

        def loader_factory(seed):
            return get_cf_dataloader(
                data_dir=data_dir,
                split="train",
                batch_size=cfg.get("train", {}).get("batch_size", 2048),
                neg_samples=cfg.get("train", {}).get("neg_samples", 1),
                num_workers=cfg.get("train", {}).get("num_workers", 0),
                seed=seed,
            )

        results = run_multi_seed(
            model_factory=model_factory,
            train_loader_factory=loader_factory,
            evaluator=evaluator,
            cfg=cfg,
            device=device,
            seeds=seeds,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )

    else:  # KG models
        dataset = build_kg_dataset(data_dir, cfg)
        evaluator = build_evaluator(dataset, cfg, device)

        def model_factory():
            return builder_fn(dataset, cfg, device)

        def loader_factory(seed):
            return get_kg_dataloader(
                data_dir=data_dir,
                split="train",
                batch_size=cfg.get("train", {}).get("batch_size", 2048),
                neg_samples=cfg.get("train", {}).get("neg_samples", 1),
                kg_type=cfg.get("dataset", {}).get("kg_type", "full"),
                num_workers=cfg.get("train", {}).get("num_workers", 0),
                seed=seed,
            )

        def kg_ds_factory():
            return build_kg_dataset(data_dir, cfg)

        results = run_kg_multi_seed(
            model_factory=model_factory,
            train_loader_factory=loader_factory,
            kg_dataset_factory=kg_ds_factory,
            evaluator=evaluator,
            cfg=cfg,
            device=device,
            seeds=seeds,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )

    # Save aggregate results
    suffix = f"_{cold_split}" if cold_split else ""
    out_path = os.path.join(result_dir, f"{model_name}{suffix}_results.json")
    with open(out_path, "w") as f:
        safe = {
            "model": model_name,
            "dataset": cfg.get("dataset", {}).get("name"),
            "cold_split": cold_split,
            "mean": results["mean"],
            "std": results["std"],
        }
        json.dump(safe, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # Cold-start eval (Cold-20 protocol for LightGCN and KG-LightGCN)
    if model_name in ("lightgcn", "kg_lightgcn", "kg_lightgcn_cl") and cold_split is None:
        _run_cold_eval(model_name, cfg, results, device, data_dir, result_dir)

    return results


def _run_cold_eval(
    model_name: str,
    cfg: dict,
    multi_seed_results: Dict,
    device: torch.device,
    data_dir: str,
    result_dir: str,
) -> None:
    """Run Cold-20 evaluation using the best checkpoint from seed=42."""
    cold_dir = os.path.join(data_dir, "cold_20")
    if not os.path.isdir(cold_dir):
        logger.warning(f"Cold-20 directory not found: {cold_dir}. Skipping cold eval.")
        return

    logger.info(f"Running Cold-20 evaluation for {model_name} …")
    # We use the seed-42 result (first seed)
    seed_result = multi_seed_results["per_seed"][0]
    checkpoint_dir = cfg.get("logging", {}).get("checkpoint_dir", "results/checkpoints")
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_seed42_best.pt")
    if not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint not found: {ckpt_path}. Skipping cold eval.")
        return

    # Rebuild model with saved weights
    dataset = (
        build_cf_dataset(data_dir, cfg)
        if model_name == "lightgcn"
        else build_kg_dataset(data_dir, cfg)
    )  # kg_lightgcn_cl also uses KGDataset
    builder_fn, _ = MODEL_BUILDERS[model_name]
    model = builder_fn(dataset, cfg, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load training data (for masking)
    train_d = dataset.read_interaction_file(os.path.join(data_dir, "train.txt"))
    test_d = dataset.read_interaction_file(os.path.join(data_dir, "test.txt"))

    cold_evaluator = ColdEvaluator(
        cold_dir=cold_dir,
        train_user2items=train_d,
        test_user2items=test_d,
        n_items=dataset.n_items,
        device=device,
    )
    cold_metrics = cold_evaluator.evaluate(model)

    logger.info(f"Cold-20 metrics ({model_name}): {cold_metrics}")
    out_path = os.path.join(result_dir, f"{model_name}_cold20_metrics.json")
    with open(out_path, "w") as f:
        json.dump(cold_metrics, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KG-LightGCN — Train & Evaluate Recommender Models"
    )
    parser.add_argument(
        "--model",
        choices=sorted(ALL_MODELS) + ["all"],
        default="lightgcn",
        help="Model to train ('all' trains every model).",
    )
    parser.add_argument(
        "--dataset",
        choices=["amazon-book", "yelp2018"],
        default="amazon-book",
    )
    parser.add_argument(
        "--cold_split",
        choices=["cold_10", "cold_20", "cold_30"],
        default=None,
        help="Train/eval on a cold split.",
    )
    parser.add_argument(
        "--kg_type",
        choices=["full", "category", "brand", "none"],
        default=None,
        help="Override KG type (entity ablation).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds to use. Default: [42, 0, 1, 2, 3].",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=None,
        help="Override n_layers (sensitivity analysis).",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Override n_workers (sensitivity analysis).",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=None,
        help="Override embedding_dim (sensitivity analysis, normally locked to 64).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Override λ regularisation (sensitivity analysis).",
    )
    parser.add_argument(
        "--base_config",
        default="configs/base.yaml",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="key=value overrides, e.g. train.learning_rate=0.0005",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    model_config_path = (
        f"configs/model/{args.model}.yaml"
        if args.model != "all" and os.path.exists(f"configs/model/{args.model}.yaml")
        else None
    )
    overrides = {}
    for item in (args.override or []):
        k, _, v = item.partition("=")
        # Try to cast value to float/int if possible
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        overrides[k] = v

    # CLI convenience overrides
    overrides["dataset.name"] = args.dataset
    if args.kg_type:
        overrides["dataset.kg_type"] = args.kg_type
    if args.n_layers:
        overrides["model.n_layers"] = args.n_layers
    if args.embedding_dim:
        overrides["model.embedding_dim"] = args.embedding_dim
    if args.weight_decay:
        overrides["train.weight_decay"] = args.weight_decay
    if args.n_workers is not None:
        overrides["train.num_workers"] = args.n_workers

    cfg = load_config(
        base_path=args.base_config,
        model_config_path=model_config_path,
        overrides=overrides,
    )

    # Seeds
    seeds = args.seeds if args.seeds else get_seeds()

    # Train
    models_to_run = list(ALL_MODELS) if args.model == "all" else [args.model]

    all_results = {}
    for model_name in models_to_run:
        # KG models require Amazon-Book
        if model_name in KG_MODELS and args.dataset == "yelp2018":
            logger.warning(f"Skipping {model_name}: KG not available for Yelp2018.")
            continue
        logger.info(f"\n{'='*70}\nRunning: {model_name}\n{'='*70}")
        results = train_model(
            model_name=model_name,
            cfg=cfg,
            seeds=seeds,
            cold_split=args.cold_split,
        )
        all_results[model_name] = results

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY (mean ± std)")
    logger.info("=" * 70)
    for model_name, res in all_results.items():
        logger.info(f"\n{model_name}:")
        for k in sorted(res.get("mean", {}).keys()):
            m = res["mean"][k]
            s = res["std"][k]
            logger.info(f"  {k}: {m:.6f} ± {s:.6f}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
"""
trainers/trainer.py
─────────────────────────────────────────────────────────────────────────────
Base trainer cho CF models (LightGCN, SimGCL).
Tối ưu cho dataset lớn:
  - eval_interval: chỉ evaluate mỗi N epoch (default=5), không phải mỗi epoch
  - log_interval: log loss mỗi epoch (nhẹ, không ảnh hưởng tốc độ)
  - Checkpoint lưu theo model/dataset/seed riêng
"""

import json
import os
import gc
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from evaluation.evaluator import Evaluator
from losses.bpr_loss import bpr_loss
from losses.contrastive_loss import infonce_loss
from utils.logger import get_logger, get_run_logger, EpochLogger, RunSummaryLogger
from utils.seed import set_seed

logger = get_logger("trainer")


class Trainer:
    """
    Base trainer cho LightGCN / SimGCL.
    Tối ưu cho dataset lớn với eval_interval để không evaluate mỗi epoch.
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        evaluator: Evaluator,
        cfg: dict,
        device: torch.device,
        checkpoint_dir: str = "results/checkpoints",
        log_dir: str = "results/logs",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.evaluator = evaluator
        self.cfg = cfg
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        train_cfg = cfg.get("train", {})
        eval_cfg  = cfg.get("eval", {})
        log_cfg   = cfg.get("logging", {})

        self.lr            = train_cfg.get("learning_rate", 1e-3)
        self.weight_decay  = train_cfg.get("weight_decay", 1e-4)
        self.epochs        = train_cfg.get("epochs", 1000)
        self.patience      = train_cfg.get("early_stopping_patience", 20)
        self.monitor_metric= train_cfg.get("early_stopping_metric", "recall@20")
        # num_workers từ config — giá trị thực tế có thể thấp hơn trên Windows
        # (safe_num_workers trong dataloader.py tự động giảm về 0 nếu cần)
        self.num_workers   = train_cfg.get("num_workers", 0)

        # eval_interval: evaluate mỗi N epoch (tiết kiệm thời gian với dataset lớn)
        self.eval_interval = eval_cfg.get("eval_interval", 10)
        # log_interval: log loss mỗi N epoch
        self.log_interval  = log_cfg.get("log_interval", 1)

        cl_cfg = cfg.get("contrastive", {})
        self.temperature = cl_cfg.get("temperature", 0.2)
        self.lambda_cl   = cl_cfg.get("lambda_cl", 0.5)

        self.model_name   = self.model.__class__.__name__.lower()
        self.dataset_name = cfg.get("dataset", {}).get("name", "unknown")

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.0
        )

        # Phát hiện loại model
        self._is_simgcl = (
            hasattr(model, "contrastive_loss")
            and not hasattr(model, "kg_forward")
            and not hasattr(model, "kg_alignment_loss")
        )

    # ── Training loop ──────────────────────────────────────────────────────────

    def train(self, seed: int = 42) -> Dict[str, Any]:
        """
        Full training loop với early stopping và eval_interval.
        Hỗ trợ resume từ checkpoint (nếu có file checkpoint với cùng model/dataset/seed).

        Returns:
            {seed, best_epoch, val_metric, test_metrics, history}
        """
        set_seed(seed)
        t_start = time.time()

        run_logger = get_run_logger(
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            seed=seed,
            base_log_dir=self.log_dir,
        )
        epoch_logger = EpochLogger(
            run_logger=run_logger,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            seed=seed,
            base_log_dir=self.log_dir,
        )
        summary_logger = RunSummaryLogger(
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            seed=seed,
            base_log_dir=self.log_dir,
        )

        run_logger.info("=" * 65)
        run_logger.info(f"  MODEL         : {self.model_name}")
        run_logger.info(f"  DATASET       : {self.dataset_name}")
        run_logger.info(f"  SEED          : {seed}")
        run_logger.info(f"  DEVICE        : {self.device}")
        run_logger.info(f"  EPOCHS        : {self.epochs}")
        run_logger.info(f"  EARLY STOPPING: {self.patience}")
        run_logger.info(f"  LR / WD       : {self.lr} / {self.weight_decay}")
        run_logger.info(f"  LAYERS        : {self.model.n_layers}")
        # Đọc num_workers thực tế từ DataLoader (có thể đã bị giảm về 0 trên Windows)
        actual_workers = getattr(self.train_loader, "num_workers", self.num_workers)
        cfg_workers    = self.num_workers
        if actual_workers != cfg_workers:
            run_logger.info(
                f"  NUM_WORKERS   : {actual_workers}  "
                f"(config={cfg_workers}, reduced: Windows/spawn không hỗ trợ sparse tensor pickle)"
            )
        else:
            run_logger.info(f"  NUM_WORKERS   : {actual_workers}")
        run_logger.info(f"  EVAL INTERVAL : every {self.eval_interval} epochs")
        run_logger.info(f"  MONITOR       : {self.monitor_metric}")
        run_logger.info(f"  PARAMS        : {self.model.parameter_count():,}")
        run_logger.info("=" * 65)

        best_metric  = -float("inf")
        best_epoch   = 0
        patience_ctr = 0
        best_state   = None
        history: List[Dict] = []
        running_loss = 0.0
        running_n    = 0
        start_epoch  = 1

        # ── Resume from checkpoint (if available) ────────────────────────────
        ckpt_path = self._get_checkpoint_path(seed)
        if os.path.exists(ckpt_path):
            try:
                resume_info = self._load_checkpoint_for_resume(ckpt_path)
                start_epoch  = resume_info["epoch"] + 1
                best_metric  = resume_info.get("best_metric", best_metric)
                best_epoch   = resume_info.get("best_epoch", resume_info["epoch"])
                patience_ctr = resume_info.get("patience_ctr", 0)
                history      = resume_info.get("history", [])
                run_logger.info(
                    f"  RESUMED from checkpoint: epoch={resume_info['epoch']}, "
                    f"best_{self.monitor_metric}={best_metric:.6f}"
                )
            except Exception as e:
                run_logger.warning(f"  Could not resume checkpoint ({e}). Starting fresh.")
                start_epoch = 1
        else:
            run_logger.info(f"  No checkpoint found at {ckpt_path}. Starting fresh.")

        for epoch in range(start_epoch, self.epochs + 1):
            t0 = time.time()
            loss = self._train_one_epoch()
            elapsed = time.time() - t0
            running_loss += loss
            running_n    += 1

            # Log loss mỗi log_interval epoch (nhẹ)
            if epoch % self.log_interval == 0:
                run_logger.info(
                    f"  [Epoch {epoch:>4}]  loss={loss:.4f}  ({elapsed:.1f}s)"
                )

            # Evaluate mỗi eval_interval epoch
            if epoch % self.eval_interval == 0:
                val_metrics  = self.evaluator.evaluate(self.model, split="val")
                monitor_val  = val_metrics.get(self.monitor_metric, 0.0)
                avg_loss     = running_loss / running_n
                running_loss = 0.0
                running_n    = 0

                epoch_logger.log(epoch, avg_loss, val_metrics, time_s=elapsed)
                history.append(
                    {"epoch": epoch, "loss": avg_loss, **val_metrics, "time_s": elapsed}
                )

                # Release unused memory (important on 32GB RAM / CPU-only)
                if self.device.type == "cpu":
                    gc.collect()
                elif self.device.type == "cuda":
                    torch.cuda.empty_cache()

                if monitor_val > best_metric:
                    best_metric  = monitor_val
                    best_epoch   = epoch
                    patience_ctr = 0
                    best_state   = deepcopy(self.model.state_dict())
                    self._save_checkpoint(
                        seed, epoch, val_metrics,
                        best_metric=best_metric,
                        best_epoch=best_epoch,
                        patience_ctr=patience_ctr,
                        history=history,
                    )
                    run_logger.info(
                        f"  *** New best @ epoch {epoch}: "
                        f"{self.monitor_metric}={best_metric:.6f} ***"
                    )
                else:
                    patience_ctr += 1
                    run_logger.info(
                        f"  [eval {epoch}]  {self.monitor_metric}={monitor_val:.6f}  "
                        f"(patience {patience_ctr}/{self.patience})"
                    )
                    # Save periodic checkpoint (every 10 eval steps) for resume
                    self._save_periodic_checkpoint(
                        seed, epoch, val_metrics,
                        best_metric=best_metric,
                        best_epoch=best_epoch,
                        patience_ctr=patience_ctr,
                        history=history,
                    )
                    if patience_ctr >= self.patience:
                        run_logger.info(
                            f"Early stopping @ epoch {epoch}. "
                            f"Best: epoch={best_epoch}, "
                            f"{self.monitor_metric}={best_metric:.6f}"
                        )
                        break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final test evaluation
        test_metrics = self.evaluator.evaluate(self.model, split="test")
        total_time   = time.time() - t_start

        run_logger.info("-" * 65)
        run_logger.info(f"FINAL TEST  (best_epoch={best_epoch})")
        for k, v in sorted(test_metrics.items()):
            run_logger.info(f"  {k:<20} = {v:.6f}")
        run_logger.info(f"Total time : {total_time:.1f}s  ({total_time/60:.1f} min)")
        run_logger.info("=" * 65)

        summary_logger.save(
            best_epoch=best_epoch,
            val_metric=best_metric,
            test_metrics=test_metrics,
            total_time_s=total_time,
        )
        epoch_logger.close()

        return {
            "seed": seed,
            "best_epoch": best_epoch,
            "val_metric": best_metric,
            "test_metrics": test_metrics,
            "history": history,
        }

    # ── One training epoch ─────────────────────────────────────────────────────

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in self.train_loader:
            users, pos_items, neg_items = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad()

            if self._is_simgcl:
                user_emb, pos_emb, neg_emb, view1, view2 = self.model(
                    users, pos_items, neg_items
                )
                loss = (
                    bpr_loss(user_emb, pos_emb, neg_emb)
                    + self.weight_decay * self.model.l2_loss(users, pos_items, neg_items)
                    + self.lambda_cl * infonce_loss(view1, view2, self.temperature)
                )
            else:
                user_emb, pos_emb, neg_emb = self.model(users, pos_items, neg_items)
                loss = (
                    bpr_loss(user_emb, pos_emb, neg_emb)
                    + self.weight_decay * self.model.l2_loss(users, pos_items, neg_items)
                )

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _get_checkpoint_path(self, seed: int) -> str:
        """Return path to the periodic (resume) checkpoint for this seed."""
        ckpt_dir = os.path.join(self.checkpoint_dir, self.dataset_name, self.model_name)
        return os.path.join(ckpt_dir, f"seed{seed}_resume.pt")

    def _get_best_checkpoint_path(self, seed: int) -> str:
        """Return path to the best-model checkpoint for this seed."""
        ckpt_dir = os.path.join(self.checkpoint_dir, self.dataset_name, self.model_name)
        return os.path.join(ckpt_dir, f"seed{seed}_best.pt")

    def _save_checkpoint(
        self,
        seed: int,
        epoch: int,
        metrics: Dict[str, float],
        best_metric: float = -float("inf"),
        best_epoch: int = 0,
        patience_ctr: int = 0,
        history: Optional[list] = None,
    ) -> None:
        """
        Save best-model checkpoint (includes full training state for resume).
        This overwrites only when a new best is found.
        """
        ckpt_dir = os.path.join(self.checkpoint_dir, self.dataset_name, self.model_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        path = self._get_best_checkpoint_path(seed)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "patience_ctr": patience_ctr,
            "history": history or [],
            "seed": seed,
            "model_name":   self.model_name,
            "dataset_name": self.dataset_name,
        }, path)
        # Also update the resume checkpoint so it always points to best
        self._save_periodic_checkpoint(
            seed, epoch, metrics,
            best_metric=best_metric,
            best_epoch=best_epoch,
            patience_ctr=patience_ctr,
            history=history,
        )

    def _save_periodic_checkpoint(
        self,
        seed: int,
        epoch: int,
        metrics: Dict[str, float],
        best_metric: float = -float("inf"),
        best_epoch: int = 0,
        patience_ctr: int = 0,
        history: Optional[list] = None,
    ) -> None:
        """
        Save periodic checkpoint (for resume after crash).
        Written every eval step so training can be resumed from the last eval epoch.
        Stores full training state: model weights, optimizer state, epoch, patience counter.
        """
        ckpt_dir = os.path.join(self.checkpoint_dir, self.dataset_name, self.model_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        path = self._get_checkpoint_path(seed)
        # Load the best model state dict if available (to restore after resume)
        best_path = self._get_best_checkpoint_path(seed)
        best_state_for_resume = None
        if os.path.exists(best_path):
            try:
                best_ckpt = torch.load(best_path, map_location=self.device)
                best_state_for_resume = best_ckpt.get("model_state_dict")
            except Exception:
                pass

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_model_state_dict": best_state_for_resume,
            "metrics": metrics,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "patience_ctr": patience_ctr,
            "history": history or [],
            "seed": seed,
            "model_name":   self.model_name,
            "dataset_name": self.dataset_name,
        }, path)

    def _load_checkpoint_for_resume(self, path: str) -> Dict[str, Any]:
        """
        Load a resume checkpoint.  Restores model weights, optimizer state,
        and training metadata (epoch, patience counter, best metric, history).

        Returns the checkpoint dict (caller uses epoch, best_metric, etc.).
        Raises on incompatible checkpoint.
        """
        ckpt = torch.load(path, map_location=self.device)

        # Validate checkpoint is for the same model/dataset
        if ckpt.get("model_name") and ckpt["model_name"] != self.model_name:
            raise ValueError(
                f"Checkpoint model_name mismatch: "
                f"expected '{self.model_name}', got '{ckpt['model_name']}'"
            )

        self.model.load_state_dict(ckpt["model_state_dict"])

        # Also restore best model state so we can keep tracking improvements
        if ckpt.get("best_model_state_dict"):
            # We keep best_state inside training loop; load here just to restore model
            pass

        if ckpt.get("optimizer_state_dict"):
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                logger.warning("Could not restore optimizer state (ignoring).")

        return ckpt

    def load_checkpoint(self, path: str) -> None:
        """Public API: load a best-model checkpoint for inference."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            f"Loaded: {path} | epoch={ckpt.get('epoch')} | metrics={ckpt.get('metrics')}"
        )


# ── Multi-seed runner ──────────────────────────────────────────────────────────

def run_multi_seed(
    model_factory,
    train_loader_factory,
    evaluator: Evaluator,
    cfg: dict,
    device: torch.device,
    seeds: Optional[List[int]] = None,
    checkpoint_dir: str = "results/checkpoints",
    log_dir: str = "results/logs",
) -> Dict[str, Any]:
    """Multi-seed training + aggregate results."""
    if seeds is None:
        seeds = [42, 0, 1, 2, 3]

    per_seed_results = []
    for seed in seeds:
        logger.info(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        model  = model_factory()
        loader = train_loader_factory(seed)
        trainer = Trainer(
            model=model,
            train_loader=loader,
            evaluator=evaluator,
            cfg=cfg,
            device=device,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )
        per_seed_results.append(trainer.train(seed=seed))

    all_metrics: Dict[str, List[float]] = {}
    for res in per_seed_results:
        for k, v in res["test_metrics"].items():
            all_metrics.setdefault(k, []).append(v)

    mean_m = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    std_m  = {k: float(np.std(v))  for k, v in all_metrics.items()}

    # Lưu multi-seed summary
    dataset_name = cfg.get("dataset", {}).get("name", "unknown")
    model_name   = model_factory().__class__.__name__.lower()
    _save_multiseed_summary(model_name, dataset_name, seeds, mean_m, std_m, log_dir)

    logger.info("\n" + "=" * 60)
    logger.info("MULTI-SEED RESULTS (mean ± std):")
    for k in sorted(mean_m):
        logger.info(f"  {k}: {mean_m[k]:.6f} ± {std_m[k]:.6f}")
    logger.info("=" * 60)

    return {"per_seed": per_seed_results, "mean": mean_m, "std": std_m}


def _save_multiseed_summary(
    model_name: str,
    dataset_name: str,
    seeds: List[int],
    mean_m: Dict[str, float],
    std_m:  Dict[str, float],
    log_dir: str,
) -> None:
    from datetime import datetime
    summary_dir = os.path.join(log_dir, dataset_name, model_name)
    os.makedirs(summary_dir, exist_ok=True)
    data = {
        "model": model_name, "dataset": dataset_name, "seeds": seeds,
        "mean": {k: round(v, 6) for k, v in mean_m.items()},
        "std":  {k: round(v, 6) for k, v in std_m.items()},
        "mean_std_str": {
            k: f"{mean_m[k]:.4f}±{std_m[k]:.4f}" for k in mean_m
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(summary_dir, "multiseed_summary.json"), "w") as f:
        json.dump(data, f, indent=2)
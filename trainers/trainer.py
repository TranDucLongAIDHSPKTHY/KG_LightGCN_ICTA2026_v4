"""
trainers/trainer.py — FIXED VERSION
Fixes vs v4:
  [FIX-TRAINER-1] Gradient clipping cho TẤT CẢ models (max_grad_norm từ config).
  [FIX-TRAINER-2] SimGCL: manual_l2_reg trong loss (khác optimizer weight_decay).
  [FIX-TRAINER-3] SimGCL item CL: user_cl + item_cl (SUM theo QRec, không avg).
  [FIX-TRAINER-4] Log patience window rõ ràng.
"""

import json, os, gc, time
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
    def __init__(self, model, train_loader, evaluator, cfg, device,
                 checkpoint_dir="results/checkpoints", log_dir="results/logs"):
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

        self.lr             = train_cfg.get("learning_rate", 1e-3)
        self.weight_decay   = train_cfg.get("weight_decay", 1e-4)
        self.epochs         = train_cfg.get("epochs", 1000)
        self.patience       = train_cfg.get("early_stopping_patience", 10)
        self.monitor_metric = train_cfg.get("early_stopping_metric", "recall@20")
        self.num_workers    = train_cfg.get("num_workers", 0)
        # [FIX-TRAINER-1] Gradient clipping
        self.max_grad_norm  = train_cfg.get("max_grad_norm", 1.0)
        # [FIX-TRAINER-2] manual_l2_reg for SimGCL
        self.manual_l2_reg  = train_cfg.get("manual_l2_reg", 0.0)

        self.eval_interval = eval_cfg.get("eval_interval", 5)
        self.log_interval  = log_cfg.get("log_interval", 1)

        cl_cfg = cfg.get("contrastive", {})
        self.temperature = cl_cfg.get("temperature", 0.2)
        self.lambda_cl   = cl_cfg.get("lambda_cl", 0.5)

        self.model_name   = self.model.__class__.__name__.lower()
        self.dataset_name = cfg.get("dataset", {}).get("name", "unknown")

        # weight_decay=0 in Adam always; manual_l2_reg added to loss directly
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0)

        self._is_simgcl = (
            hasattr(model, "contrastive_loss")
            and not hasattr(model, "kg_forward")
            and not hasattr(model, "kg_alignment_loss")
        )

    def train(self, seed: int = 42) -> Dict[str, Any]:
        set_seed(seed)
        t_start = time.time()

        run_logger = get_run_logger(self.model_name, self.dataset_name, seed,
                                    base_log_dir=self.log_dir)
        epoch_logger = EpochLogger(run_logger, self.model_name, self.dataset_name,
                                   seed, base_log_dir=self.log_dir)
        summary_logger = RunSummaryLogger(self.model_name, self.dataset_name, seed,
                                          base_log_dir=self.log_dir)

        patience_window = self.patience * self.eval_interval

        run_logger.info("=" * 65)
        run_logger.info(f"  MODEL          : {self.model_name}")
        run_logger.info(f"  DATASET        : {self.dataset_name}")
        run_logger.info(f"  SEED           : {seed}")
        run_logger.info(f"  DEVICE         : {self.device}")
        run_logger.info(f"  EPOCHS         : {self.epochs}")
        run_logger.info(f"  EARLY STOPPING : patience={self.patience}, window={patience_window} epochs")
        run_logger.info(f"  LR             : {self.lr}")
        run_logger.info(f"  WEIGHT_DECAY   : {self.weight_decay} (optimizer; =0 for SimGCL)")
        if self.manual_l2_reg > 0:
            run_logger.info(f"  MANUAL_L2_REG  : {self.manual_l2_reg} (added to loss directly)")
        run_logger.info(f"  MAX_GRAD_NORM  : {self.max_grad_norm}")
        run_logger.info(f"  LAYERS         : {self.model.n_layers}")
        run_logger.info(f"  EVAL INTERVAL  : every {self.eval_interval} epochs")
        run_logger.info(f"  MONITOR        : {self.monitor_metric}")
        run_logger.info(f"  PARAMS         : {self.model.parameter_count():,}")
        if self._is_simgcl:
            run_logger.info(f"  LAMBDA_CL      : {self.lambda_cl}")
            run_logger.info(f"  TEMPERATURE    : {self.temperature}")
            run_logger.info(f"  APPLY_ITEM_CL  : {getattr(self.model, 'apply_item_cl', False)}")
        run_logger.info("=" * 65)

        best_metric  = -float("inf")
        best_epoch   = 0
        patience_ctr = 0
        best_state   = None
        history: List[Dict] = []
        running_loss = 0.0
        running_n    = 0
        start_epoch  = 1
        epoch_times: List[float] = []

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
                    f"  RESUMED: epoch={resume_info['epoch']}, "
                    f"best_{self.monitor_metric}={best_metric:.6f}, "
                    f"patience={patience_ctr}/{self.patience}"
                )
            except Exception as e:
                run_logger.warning(f"  Cannot resume ({e}). Starting fresh.")
                start_epoch = 1
        else:
            run_logger.info(f"  No checkpoint at {ckpt_path}. Starting fresh.")

        for epoch in range(start_epoch, self.epochs + 1):
            t0 = time.time()
            loss = self._train_one_epoch()
            elapsed = time.time() - t0
            running_loss += loss
            running_n    += 1

            if epoch % self.log_interval == 0:
                epoch_times.append(elapsed)
                avg_t = sum(epoch_times[-10:]) / len(epoch_times[-10:])
                eta = time.strftime("%H:%M:%S", time.gmtime(avg_t * (self.epochs - epoch)))
                run_logger.info(
                    f"  [Epoch {epoch}/{self.epochs}] loss={loss:.4f} | "
                    f"{elapsed:.1f}s | ETA={eta}"
                )

            if epoch % self.eval_interval == 0:
                val_metrics = self.evaluator.evaluate(self.model, split="val")
                monitor_val = val_metrics.get(self.monitor_metric, 0.0)
                avg_loss    = running_loss / running_n
                running_loss = 0.0; running_n = 0

                epoch_logger.log(epoch, avg_loss, val_metrics, time_s=elapsed)
                history.append({"epoch": epoch, "loss": avg_loss, **val_metrics, "time_s": elapsed})

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

                if monitor_val > best_metric:
                    best_metric  = monitor_val
                    best_epoch   = epoch
                    patience_ctr = 0
                    best_state   = deepcopy(self.model.state_dict())
                    self._save_checkpoint(seed, epoch, val_metrics,
                                          best_metric=best_metric,
                                          best_epoch=best_epoch,
                                          patience_ctr=patience_ctr,
                                          history=history)
                    run_logger.info(
                        f"  *** New best @ epoch {epoch}: "
                        f"{self.monitor_metric}={best_metric:.6f} ***"
                    )
                else:
                    patience_ctr += 1
                    no_imp = patience_ctr * self.eval_interval
                    run_logger.info(
                        f"  [eval {epoch}]  {self.monitor_metric}={monitor_val:.6f}  "
                        f"(no improve: {no_imp}/{patience_window} epochs, "
                        f"patience {patience_ctr}/{self.patience})"
                    )
                    self._save_periodic_checkpoint(seed, epoch, val_metrics,
                                                   best_metric=best_metric,
                                                   best_epoch=best_epoch,
                                                   patience_ctr=patience_ctr,
                                                   history=history)
                    if patience_ctr >= self.patience:
                        run_logger.info(
                            f"Early stopping @ epoch {epoch}. "
                            f"Best: epoch={best_epoch}, "
                            f"{self.monitor_metric}={best_metric:.6f}"
                        )
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        test_metrics = self.evaluator.evaluate(self.model, split="test")
        total_time   = time.time() - t_start

        run_logger.info("-" * 65)
        run_logger.info(f"FINAL TEST  (best_epoch={best_epoch})")
        for k, v in sorted(test_metrics.items()):
            run_logger.info(f"  {k:<20} = {v:.6f}")
        run_logger.info(f"Total time : {total_time:.1f}s  ({total_time/60:.1f} min)")
        run_logger.info("=" * 65)

        summary_logger.save(best_epoch=best_epoch, val_metric=best_metric,
                            test_metrics=test_metrics, total_time_s=total_time)
        epoch_logger.close()

        return {"seed": seed, "best_epoch": best_epoch, "val_metric": best_metric,
                "test_metrics": test_metrics, "history": history}

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in self.train_loader:
            users, pos_items, neg_items = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad()

            if self._is_simgcl:
                output = self.model(users, pos_items, neg_items)
                user_emb, pos_emb, neg_emb = output[0], output[1], output[2]
                view1, view2 = output[3], output[4]

                rec_loss = bpr_loss(user_emb, pos_emb, neg_emb)
                # [FIX-TRAINER-3] user CL
                cl_total = infonce_loss(view1, view2, self.temperature)

                # [FIX-TRAINER-3] item CL: SUM với user CL (QRec protocol)
                if len(output) == 7:
                    item_cl = infonce_loss(output[5], output[6], self.temperature)
                    cl_total = cl_total + item_cl  # SUM không phải /2

                # [FIX-TRAINER-2] manual_l2_reg trực tiếp vào loss
                # (optimizer weight_decay=0, paper dùng explicit L2 penalty)
                l2_loss = self.model.l2_loss(users, pos_items, neg_items)
                loss = rec_loss + self.lambda_cl * cl_total + self.manual_l2_reg * l2_loss

            else:
                # LightGCN và các CF models khác: weight_decay qua loss
                user_emb, pos_emb, neg_emb = self.model(users, pos_items, neg_items)
                loss = (bpr_loss(user_emb, pos_emb, neg_emb)
                        + self.weight_decay * self.model.l2_loss(users, pos_items, neg_items))

            loss.backward()

            # [FIX-TRAINER-1] Gradient clipping — áp dụng cho TẤT CẢ models
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.max_grad_norm)

            self.optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _get_checkpoint_path(self, seed):
        d = os.path.join(self.checkpoint_dir, self.dataset_name, self.model_name)
        return os.path.join(d, f"seed{seed}_resume.pt")

    def _get_best_checkpoint_path(self, seed):
        d = os.path.join(self.checkpoint_dir, self.dataset_name, self.model_name)
        return os.path.join(d, f"seed{seed}_best.pt")

    def _save_checkpoint(self, seed, epoch, metrics, best_metric=-float("inf"),
                         best_epoch=0, patience_ctr=0, history=None):
        d = os.path.join(self.checkpoint_dir, self.dataset_name, self.model_name)
        os.makedirs(d, exist_ok=True)
        torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "metrics": metrics, "best_metric": best_metric,
                    "best_epoch": best_epoch, "patience_ctr": patience_ctr,
                    "history": history or [], "seed": seed,
                    "model_name": self.model_name,
                    "dataset_name": self.dataset_name},
                   self._get_best_checkpoint_path(seed))
        self._save_periodic_checkpoint(seed, epoch, metrics, best_metric=best_metric,
                                       best_epoch=best_epoch, patience_ctr=patience_ctr,
                                       history=history)

    def _save_periodic_checkpoint(self, seed, epoch, metrics, best_metric=-float("inf"),
                                   best_epoch=0, patience_ctr=0, history=None):
        d = os.path.join(self.checkpoint_dir, self.dataset_name, self.model_name)
        os.makedirs(d, exist_ok=True)
        best_state_for_resume = None
        best_path = self._get_best_checkpoint_path(seed)
        if os.path.exists(best_path):
            try:
                best_state_for_resume = torch.load(best_path,
                    map_location=self.device).get("model_state_dict")
            except Exception:
                pass
        torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_model_state_dict": best_state_for_resume,
                    "metrics": metrics, "best_metric": best_metric,
                    "best_epoch": best_epoch, "patience_ctr": patience_ctr,
                    "history": history or [], "seed": seed,
                    "model_name": self.model_name,
                    "dataset_name": self.dataset_name},
                   self._get_checkpoint_path(seed))

    def _load_checkpoint_for_resume(self, path):
        ckpt = torch.load(path, map_location=self.device)
        if ckpt.get("model_name") and ckpt["model_name"] != self.model_name:
            raise ValueError(f"model_name mismatch: {ckpt['model_name']} vs {self.model_name}")
        self.model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                logger.warning("Could not restore optimizer state.")
        return ckpt

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded: {path} | epoch={ckpt.get('epoch')} | metrics={ckpt.get('metrics')}")


def run_multi_seed(model_factory, train_loader_factory, evaluator, cfg, device,
                   seeds=None, checkpoint_dir="results/checkpoints",
                   log_dir="results/logs"):
    if seeds is None:
        seeds = [42, 0, 1, 2, 3]
    per_seed_results = []
    for seed in seeds:
        logger.info(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        model  = model_factory()
        loader = train_loader_factory(seed)
        trainer = Trainer(model=model, train_loader=loader, evaluator=evaluator,
                          cfg=cfg, device=device, checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir)
        per_seed_results.append(trainer.train(seed=seed))

    all_metrics: Dict[str, List[float]] = {}
    for res in per_seed_results:
        for k, v in res["test_metrics"].items():
            all_metrics.setdefault(k, []).append(v)

    mean_m = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    std_m  = {k: float(np.std(v))  for k, v in all_metrics.items()}
    dataset_name = cfg.get("dataset", {}).get("name", "unknown")
    model_name   = model_factory().__class__.__name__.lower()
    _save_multiseed_summary(model_name, dataset_name, seeds, mean_m, std_m, log_dir)

    logger.info("\n" + "=" * 60)
    logger.info("MULTI-SEED RESULTS (mean ± std):")
    for k in sorted(mean_m):
        logger.info(f"  {k}: {mean_m[k]:.6f} ± {std_m[k]:.6f}")
    logger.info("=" * 60)
    return {"per_seed": per_seed_results, "mean": mean_m, "std": std_m}


def _save_multiseed_summary(model_name, dataset_name, seeds, mean_m, std_m, log_dir):
    from datetime import datetime
    d = os.path.join(log_dir, dataset_name, model_name)
    os.makedirs(d, exist_ok=True)
    data = {"model": model_name, "dataset": dataset_name, "seeds": seeds,
            "mean": {k: round(v, 6) for k, v in mean_m.items()},
            "std":  {k: round(v, 6) for k, v in std_m.items()},
            "mean_std_str": {k: f"{mean_m[k]:.4f}±{std_m[k]:.4f}" for k in mean_m},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(d, "multiseed_summary.json"), "w") as f:
        json.dump(data, f, indent=2)
"""
trainers/kg_trainer.py
─────────────────────────────────────────────────────────────────────────────
Trainer for Knowledge Graph models: KGAT, KGCL, KG-LightGCN.

Fixes vs previous version:
  [BUG-C1-FIX] self.lambda_cl không được khai báo trong __init__.
               _kgcl_step() tham chiếu self.lambda_cl → AttributeError âm thầm
               (Python resolve về base class rồi crash hoặc dùng giá trị sai).
               Fix: khai báo tường minh từ cfg["contrastive"]["lambda_cl"].

  [BUG-C2-FIX] _kgcl_step(): KGCL forward giờ trả về 7 tensors
               (user_emb, pos, neg, u1, u2, i1, i2). Unpack đúng và tính
               cả user_cl + item_cl như paper Eq.(9).

  [BUG-K4-FIX] KGAT: truyền kg_n_layers vào model constructor qua build_kgat
               (đã sửa trong configs/model/kgat.yaml và main.py).
               Ở đây: entity_emb cache vẫn giữ như cũ (tính 1 lần/epoch).
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
from tqdm import tqdm

from datasets.kg_dataset import KGDataset
from evaluation.evaluator import Evaluator
from losses.bpr_loss import bpr_loss, kg_bpr_loss
from losses.contrastive_loss import infonce_loss
from trainers.trainer import Trainer
from utils.logger import get_logger, get_run_logger, EpochLogger, RunSummaryLogger
from utils.seed import set_seed

logger = get_logger("kg_trainer")


class KGTrainer(Trainer):
    """
    Trainer for KG-enhanced recommender models.

    Extends Trainer._train_one_epoch() to handle:
      - KG BPR loss (KGAT, via kg_forward)
      - Contrastive loss (KGCL) — user + item CL
      - KG alignment loss (KG-LightGCN, via kg_alignment_loss)
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        kg_dataset: KGDataset,
        evaluator: Evaluator,
        cfg: dict,
        device: torch.device,
        checkpoint_dir: str = "results/checkpoints",
        log_dir: str = "results/logs",
    ) -> None:
        super().__init__(
            model=model,
            train_loader=train_loader,
            evaluator=evaluator,
            cfg=cfg,
            device=device,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )
        self.kg_dataset = kg_dataset

        # Detect model type (order matters: most specific first)
        self._is_kgat = hasattr(model, "kg_forward")
        self._is_kg_lightgcn_cl = (
            hasattr(model, "contrastive_loss")
            and hasattr(model, "kg_alignment_loss")
            and hasattr(model, "cl_temp")
        )
        self._is_kgcl = (
            hasattr(model, "contrastive_loss")
            and hasattr(model, "set_kg_norm_adj")
            and not self._is_kg_lightgcn_cl
        )
        self._is_kg_lightgcn = (
            hasattr(model, "kg_alignment_loss")
            and not self._is_kg_lightgcn_cl
        )

        # KG loss weight
        model_cfg = cfg.get("model", {})
        self.lambda_kg = float(model_cfg.get("kg_reg", 1e-5))

        # [BUG-C1-FIX] Khai báo tường minh lambda_cl từ contrastive config
        # Version cũ không khai báo → self.lambda_cl không tồn tại →
        # _kgcl_step() crash hoặc dùng giá trị từ base class (0.5 thay vì 0.1)
        cl_cfg = cfg.get("contrastive", {})
        self.lambda_cl = float(cl_cfg.get("lambda_cl", 0.1))

        self.kg_steps_per_cf = 1
        self._cached_entity_emb: Optional[torch.Tensor] = None

    # ── Override epoch training ───────────────────────────────────────────────

    def _train_one_epoch(self) -> float:
        """
        Training epoch for KG models.
        """
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        if self._is_kgat:
            self._free_entity_emb_cache()
            with torch.no_grad():
                self._cached_entity_emb = (
                    self.model._compute_entity_embeddings().detach()
                )
        else:
            self._cached_entity_emb = None

        # KGCL: rebuild augmented adj matrices once per epoch
        if self._is_kgcl and hasattr(self.model, "refresh_augmented_views"):
            with torch.no_grad():
                self.model.refresh_augmented_views()

        for batch in self.train_loader:
            users, pos_items, neg_items = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad()

            if self._is_kgat:
                loss = self._kgat_step(users, pos_items, neg_items)
            elif self._is_kg_lightgcn_cl:
                loss = self._kg_lightgcn_cl_step(users, pos_items, neg_items)
            elif self._is_kgcl:
                loss = self._kgcl_step(users, pos_items, neg_items)
            elif self._is_kg_lightgcn:
                loss = self._kg_lightgcn_step(users, pos_items, neg_items)
            else:
                user_emb, pos_emb, neg_emb = self.model(users, pos_items, neg_items)
                rec_loss = bpr_loss(user_emb, pos_emb, neg_emb)
                reg_loss = self.model.l2_loss(users, pos_items, neg_items)
                loss = rec_loss + self.weight_decay * reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        if self._is_kgat:
            self._free_entity_emb_cache()

        return total_loss / max(n_batches, 1)

    # ── OOM helper ────────────────────────────────────────────────────────────

    def _free_entity_emb_cache(self) -> None:
        if self._cached_entity_emb is not None:
            del self._cached_entity_emb
            self._cached_entity_emb = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Model-specific step functions ─────────────────────────────────────────

    def _kgat_step(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """KGAT: L = BPR(CF) + BPR(KG/TransR) + λ·L2"""
        user_emb, pos_emb, neg_emb = self.model(
            users, pos_items, neg_items,
            precomputed_entity_emb=self._cached_entity_emb,
        )
        cf_loss = bpr_loss(user_emb, pos_emb, neg_emb)
        l2      = self.model.l2_loss(users, pos_items, neg_items)

        if self.kg_dataset.kg_triples is not None:
            batch_size = len(users)
            triples = self.kg_dataset.sample_kg_triples(batch_size)
            if triples is not None:
                heads, rels, t_pos, t_neg = triples
                heads   = torch.tensor(heads,  dtype=torch.long, device=self.device)
                rels    = torch.tensor(rels,   dtype=torch.long, device=self.device)
                t_pos_t = torch.tensor(t_pos,  dtype=torch.long, device=self.device)
                t_neg_t = torch.tensor(t_neg,  dtype=torch.long, device=self.device)
                pos_score, neg_score = self.model.kg_forward(heads, rels, t_pos_t, t_neg_t)
                kg_loss = kg_bpr_loss(pos_score, neg_score)
            else:
                h, r, t_pos_s, t_neg_s = self.kg_dataset.sample_kg_triple()
                heads   = torch.tensor([h],       dtype=torch.long, device=self.device)
                rels    = torch.tensor([r],        dtype=torch.long, device=self.device)
                t_pos_t = torch.tensor([t_pos_s],  dtype=torch.long, device=self.device)
                t_neg_t = torch.tensor([t_neg_s],  dtype=torch.long, device=self.device)
                pos_score, neg_score = self.model.kg_forward(heads, rels, t_pos_t, t_neg_t)
                kg_loss = kg_bpr_loss(pos_score, neg_score)
        else:
            kg_loss = torch.tensor(0.0, device=self.device)

        return cf_loss + self.lambda_kg * kg_loss + self.weight_decay * l2

    def _kgcl_step(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """
        KGCL: L = BPR + λ_cl·(UserInfoNCE + ItemInfoNCE) + λ·L2

        [BUG-C1-FIX] Dùng self.lambda_cl (đã khai báo đúng trong __init__).
        [BUG-C2-FIX] Unpack 7 tensors, tính cả user_cl + item_cl.
        """
        # [BUG-C2-FIX] KGCL forward giờ trả về 7 tensors
        (
            user_emb, pos_emb, neg_emb,
            u1, u2,
            i1, i2,
        ) = self.model(users, pos_items, neg_items)

        cf_loss = bpr_loss(user_emb, pos_emb, neg_emb)
        l2      = self.model.l2_loss(users, pos_items, neg_items)

        # [BUG-C2-FIX] Tính cả user CL và item CL
        user_cl = self.model.contrastive_loss(u1, u2)
        item_cl = self.model.contrastive_loss(i1, i2)
        cl_loss = (user_cl + item_cl) / 2.0

        # [BUG-C1-FIX] self.lambda_cl = 0.1 (KGCL paper), không phải 0.5
        return cf_loss + self.lambda_cl * cl_loss + self.weight_decay * l2

    def _kg_lightgcn_cl_step(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """
        KG-LightGCN-CL: L = BPR + λ_cl*(UserCL+ItemCL) + λ_kg*KGAlign + λ_reg*L2
        """
        (
            user_emb, pos_emb, neg_emb,
            user_cf, user_kg,
            item_cf, item_kg,
        ) = self.model(users, pos_items, neg_items)

        cf_loss = bpr_loss(user_emb, pos_emb, neg_emb)

        user_cl = self.model.contrastive_loss(user_cf, user_kg)
        item_cl = self.model.contrastive_loss(item_cf, item_kg)
        cl_loss = (user_cl + item_cl) / 2.0

        align_loss = self.model.kg_alignment_loss()
        l2         = self.model.l2_loss(users, pos_items, neg_items)

        return (
            cf_loss
            + self.lambda_cl * cl_loss
            + self.lambda_kg * align_loss
            + self.weight_decay * l2
        )

    def _kg_lightgcn_step(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """KG-LightGCN Base: L = BPR + λ_kg·KG_align + λ·L2"""
        user_emb, pos_emb, neg_emb = self.model(users, pos_items, neg_items)
        cf_loss    = bpr_loss(user_emb, pos_emb, neg_emb)
        l2         = self.model.l2_loss(users, pos_items, neg_items)
        align_loss = self.model.kg_alignment_loss()
        return cf_loss + self.lambda_kg * align_loss + self.weight_decay * l2


# ── Multi-seed for KG models ──────────────────────────────────────────────────

def run_kg_multi_seed(
    model_factory,
    train_loader_factory,
    kg_dataset_factory,
    evaluator: Evaluator,
    cfg: dict,
    device: torch.device,
    seeds: Optional[List[int]] = None,
    checkpoint_dir: str = "results/checkpoints",
    log_dir: str = "results/logs",
) -> Dict[str, Any]:
    """Multi-seed training for KG models."""
    if seeds is None:
        seeds = [42, 0, 1, 2, 3]

    per_seed_results = []
    for seed in seeds:
        logger.info(f"\n{'='*60}\nKG Seed {seed}\n{'='*60}")
        set_seed(seed)
        model      = model_factory()
        loader     = train_loader_factory(seed)
        kg_dataset = kg_dataset_factory()

        trainer = KGTrainer(
            model=model,
            train_loader=loader,
            kg_dataset=kg_dataset,
            evaluator=evaluator,
            cfg=cfg,
            device=device,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )
        result = trainer.train(seed=seed)
        per_seed_results.append(result)

        del trainer, model, loader, kg_dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_metrics = {}
    for result in per_seed_results:
        for k, v in result["test_metrics"].items():
            all_metrics.setdefault(k, []).append(v)

    mean_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    std_metrics  = {k: float(np.std(v))  for k, v in all_metrics.items()}

    logger.info("\n" + "=" * 60)
    logger.info("KG MULTI-SEED RESULTS (mean ± std):")
    for k in sorted(mean_metrics.keys()):
        logger.info(f"  {k}: {mean_metrics[k]:.6f} ± {std_metrics[k]:.6f}")
    logger.info("=" * 60)

    return {"per_seed": per_seed_results, "mean": mean_metrics, "std": std_metrics}
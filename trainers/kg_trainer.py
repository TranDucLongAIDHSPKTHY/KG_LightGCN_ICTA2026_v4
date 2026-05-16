"""
trainers/kg_trainer.py
─────────────────────────────────────────────────────────────────────────────
Trainer for Knowledge Graph models: KGAT, KGCL, KG-LightGCN.

Extends the base Trainer with:
  - Alternating KG + CF training (KGAT style)
  - KG BPR loss for TransR embedding (KGAT)
  - Contrastive loss with KG augmentation (KGCL)
  - KG alignment loss (KG-LightGCN)

[OOM-FIX] Changes vs original:
  - _compute_entity_embeddings() is now chunked in kgat.py (primary fix).
  - Here: torch.cuda.empty_cache() + gc.collect() BEFORE caching entity emb,
    not after, so peak VRAM is minimised at the most expensive moment.
  - _cached_entity_emb is explicitly deleted and cache cleared after each
    epoch to avoid VRAM accumulation across epochs.
  - Added `entity_emb_chunk_size` passthrough from cfg → KGAT constructor.
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
      - Contrastive loss (KGCL)
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
        """
        Args:
            model:          KG model (KGAT | KGCL | KGLightGCN).
            train_loader:   DataLoader for CF (user, pos, neg) batches.
            kg_dataset:     KGDataset instance for KG triple sampling.
            evaluator:      Shared Evaluator.
            cfg:            Config dict.
            device:         Training device.
            checkpoint_dir: Checkpoint save directory.
            log_dir:        Log directory.
        """
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
        # KGLightGCNCL: has contrastive_loss + kg_alignment_loss + cl_temp
        self._is_kg_lightgcn_cl = (
            hasattr(model, "contrastive_loss")
            and hasattr(model, "kg_alignment_loss")
            and hasattr(model, "cl_temp")
        )
        # KGCL: has contrastive_loss + set_kg_norm_adj, but NOT cl_temp
        self._is_kgcl = (
            hasattr(model, "contrastive_loss")
            and hasattr(model, "set_kg_norm_adj")
            and not self._is_kg_lightgcn_cl
        )
        # Base KGLightGCN: has kg_alignment_loss but NOT contrastive_loss
        self._is_kg_lightgcn = (
            hasattr(model, "kg_alignment_loss")
            and not self._is_kg_lightgcn_cl
        )

        # KG loss weight (from model config or default)
        model_cfg = cfg.get("model", {})
        self.lambda_kg = float(model_cfg.get("kg_reg", 1e-5))
        self.kg_steps_per_cf = 1  # alternate: 1 KG step per CF step

        # [OOM-FIX] Epoch-level cache for KGAT entity embeddings.
        # Initialised to None; set once at start of each epoch; deleted at end.
        self._cached_entity_emb: Optional[torch.Tensor] = None

    # ── Override epoch training ───────────────────────────────────────────────

    def _train_one_epoch(self) -> float:
        """
        Training epoch for KG models.
        Combines CF BPR loss with a model-specific KG loss term.

        [OOM-FIX] KGAT entity embeddings:
          - Computed ONCE per epoch (not per-batch) to avoid 1.4M-triple
            recomputation every forward() call.
          - empty_cache() is called BEFORE the expensive computation so that
            all stale activations from the previous epoch are freed first.
          - The cache is deleted and empty_cache() called AGAIN at epoch end
            to reclaim VRAM before evaluation or the next epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        if self._is_kgat:
            # [OOM-FIX] Free stale tensors BEFORE the expensive KG aggregation.
            # Doing it after (as in original) means peak VRAM = old + new.
            self._free_entity_emb_cache()

            with torch.no_grad():
                # _compute_entity_embeddings() now runs in chunks internally
                # (see kgat.py [OOM-FIX]), so this no longer OOMs on 4–16 GB GPUs.
                self._cached_entity_emb = (
                    self.model._compute_entity_embeddings().detach()
                )
        else:
            self._cached_entity_emb = None

        # KGCL: rebuild augmented adj matrices ONCE per epoch (not per batch).
        if self._is_kgcl and hasattr(self.model, "refresh_augmented_views"):
            with torch.no_grad():
                self.model.refresh_augmented_views()

        for batch in self.train_loader:
            users, pos_items, neg_items = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad()

            # ── KGAT: alternate KG embedding training ────────────────────────
            if self._is_kgat:
                loss = self._kgat_step(users, pos_items, neg_items)

            # ── KG-LightGCN-CL: BPR + Cross-view CL + KG alignment ───────────
            elif self._is_kg_lightgcn_cl:
                loss = self._kg_lightgcn_cl_step(users, pos_items, neg_items)

            # ── KGCL: BPR + KG augmented contrastive loss ────────────────────
            elif self._is_kgcl:
                loss = self._kgcl_step(users, pos_items, neg_items)

            # ── KG-LightGCN (Base): BPR + KG alignment ───────────────────────
            elif self._is_kg_lightgcn:
                loss = self._kg_lightgcn_step(users, pos_items, neg_items)

            else:
                # Fallback to base CF training
                user_emb, pos_emb, neg_emb = self.model(users, pos_items, neg_items)
                rec_loss = bpr_loss(user_emb, pos_emb, neg_emb)
                reg_loss = self.model.l2_loss(users, pos_items, neg_items)
                loss = rec_loss + self.weight_decay * reg_loss

            loss.backward()
            # Gradient clipping (helps with KG training stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        # [OOM-FIX] Free entity emb cache at epoch end so VRAM is available
        # for evaluation (full-ranking over all items needs spare VRAM too).
        if self._is_kgat:
            self._free_entity_emb_cache()

        return total_loss / max(n_batches, 1)

    # ── OOM helper ────────────────────────────────────────────────────────────

    def _free_entity_emb_cache(self) -> None:
        """
        [OOM-FIX] Delete the epoch-level entity embedding cache and free VRAM.
        Safe to call even when cache is already None.
        """
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
        """
        KGAT training step:
          L = BPR(CF) + BPR(KG/TransR) + λ·L2

        Samples a batch of KG triples (same size as CF batch) for efficiency.
        Uses epoch-level cached entity embeddings to avoid redundant KG aggregation.
        """
        # CF loss — pass precomputed entity_emb (set at epoch start)
        user_emb, pos_emb, neg_emb = self.model(
            users, pos_items, neg_items,
            precomputed_entity_emb=self._cached_entity_emb,
        )
        cf_loss = bpr_loss(user_emb, pos_emb, neg_emb)
        l2      = self.model.l2_loss(users, pos_items, neg_items)

        # KG BPR loss (TransR) — sample a batch of triples
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
                # Fallback: single triple
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
        KGCL training step:
          L = BPR + λ_cl·InfoNCE + λ·L2
        """
        user_emb, pos_emb, neg_emb, view1, view2 = self.model(
            users, pos_items, neg_items
        )
        cf_loss = bpr_loss(user_emb, pos_emb, neg_emb)
        l2      = self.model.l2_loss(users, pos_items, neg_items)
        cl_loss = self.model.contrastive_loss(view1, view2)
        return cf_loss + self.lambda_cl * cl_loss + self.weight_decay * l2

    def _kg_lightgcn_cl_step(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """
        KG-LightGCN-CL training step (Biến thể 2 - Enhanced/Proposed):
          L = BPR + λ_cl*(UserCL + ItemCL) + λ_kg*KGAlign + λ_reg*L2

        Forward trả về 7 tensors:
          user_emb, pos_emb, neg_emb  -- cho BPR
          user_cf,  user_kg           -- user contrastive views
          item_cf,  item_kg           -- item contrastive views
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
        """
        KG-LightGCN training step:
          L = BPR + λ_kg·KG_align + λ·L2
        """
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
    """
    Multi-seed training for KG models.

    Args:
        model_factory:          Callable() → model.
        train_loader_factory:   Callable(seed) → DataLoader.
        kg_dataset_factory:     Callable() → KGDataset.
        evaluator:              Shared Evaluator.
        cfg:                    Config dict.
        device:                 Training device.
        seeds:                  List of seeds.
        checkpoint_dir:         Checkpoint directory.
        log_dir:                Log directory.

    Returns:
        {per_seed, mean, std}
    """
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

        # [OOM-FIX] Aggressively free resources between seeds to avoid
        # accumulation of GPU tensors across seeds in the same process.
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
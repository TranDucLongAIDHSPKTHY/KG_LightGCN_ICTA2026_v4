"""
models/simgcl.py
─────────────────────────────────────────────────────────────────────────────
SimGCL: Are Graph Augmentations Necessary? Simple Graph Contrastive Learning
for Recommendation.
Yu et al., SIGIR 2022 — https://arxiv.org/abs/2112.08679

Fixes vs previous version:
  [BUG-S1-FIX] forward() luôn trả về tuple cố định 7 phần tử:
               (user_emb, pos_emb, neg_emb, u1, u2, i1, i2)
               khi apply_item_cl=True, hoặc 5 phần tử khi False.
               Trainer bây giờ unpack an toàn bằng cách kiểm tra len().

  [BUG-S2-NOTE] Paper SimGCL dùng CÙNG perturbed view cho BPR và CL:
               - Thực tế QRec/paper code: BPR dùng clean embedding,
                 2 CL views dùng perturb=True. Giữ nguyên hành vi này.
               - Không share clean embedding với CL view để tránh
                 information leak (clean + perturbed ≠ hai independent views).

  Giữ nguyên [BUG-1] fix: noise = (rand*2-1)*eps (uniform, no normalize).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel


class SimGCL(BaseModel):
    """
    SimGCL: Simple Graph Contrastive Learning for Recommendation.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        eps: float = 0.1,
        temperature: float = 0.2,
        lambda_cl: float = 0.5,
        apply_item_cl: bool = False,
        norm_adj: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(n_users, n_items, embedding_dim, device)
        self.n_layers      = n_layers
        self.eps           = eps
        self.temperature   = temperature
        self.lambda_cl     = lambda_cl
        self.apply_item_cl = apply_item_cl

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        if norm_adj is not None:
            self.register_buffer("norm_adj", norm_adj)
        else:
            self.norm_adj = None

        self._init_weights()

    # ── Graph propagation ─────────────────────────────────────────────────────

    def _propagate(
        self, perturb: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LightGCN-style propagation with optional per-layer uniform noise.

        [BUG-1-FIX] Uniform noise per element: ε ~ Uniform(-eps, eps).
        """
        _dev = self.user_embedding.weight.device
        adj  = self.norm_adj.to(_dev)
        E0   = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        E_k  = E0
        acc  = E0.clone()

        for _ in range(self.n_layers):
            E_k = torch.sparse.mm(adj, E_k)
            if perturb:
                noise = (torch.rand_like(E_k) * 2 - 1) * self.eps
                E_k   = E_k + noise
            acc = acc + E_k

        E_final = acc / (self.n_layers + 1)
        return E_final[: self.n_users], E_final[self.n_users :]

    # ── BaseModel interface ───────────────────────────────────────────────────

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Returns BPR embeddings + augmented views for CL loss.

        [BUG-S1-FIX] Luôn trả về tuple có kích thước nhất quán:
          apply_item_cl=False (default): (user_emb, pos, neg, u1, u2)        — 5 tensors
          apply_item_cl=True:            (user_emb, pos, neg, u1, u2, i1, i2) — 7 tensors

        Trainer._train_one_epoch() kiểm tra len() trước khi unpack.
        """
        # Clean propagation cho BPR
        user_emb, item_emb = self._propagate(perturb=False)
        u_emb = user_emb[users]
        p_emb = item_emb[pos_items]
        n_emb = item_emb[neg_items]

        # Hai augmented views (perturbed, independent)
        u1, i1 = self._propagate(perturb=True)
        u2, i2 = self._propagate(perturb=True)

        if self.apply_item_cl:
            return u_emb, p_emb, n_emb, u1[users], u2[users], i1[pos_items], i2[pos_items]
        return u_emb, p_emb, n_emb, u1[users], u2[users]

    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return clean (non-perturbed) embeddings for evaluation."""
        return self._propagate(perturb=False)

    def contrastive_loss(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric InfoNCE loss between two views."""
        v1  = F.normalize(view1, dim=-1)
        v2  = F.normalize(view2, dim=-1)
        sim = torch.matmul(v1, v2.T) / self.temperature
        labels = torch.arange(len(v1), device=v1.device)
        loss   = F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)
        return loss / 2

    def l2_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        u0 = self.user_embedding(users)
        p0 = self.item_embedding(pos_items)
        n0 = self.item_embedding(neg_items)
        return (u0.norm(2).pow(2) + p0.norm(2).pow(2) + n0.norm(2).pow(2)) / (2 * len(users))

    def set_adj(self, norm_adj: torch.Tensor) -> None:
        self.register_buffer("norm_adj", norm_adj)
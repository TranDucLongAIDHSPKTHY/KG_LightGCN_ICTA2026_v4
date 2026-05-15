# _lazy_device_fixed_
"""
models/simgcl.py
─────────────────────────────────────────────────────────────────────────────
SimGCL: Are Graph Augmentations Necessary?  Simple Graph Contrastive Learning
for Recommendation.
Yu et al., SIGIR 2022 — https://arxiv.org/abs/2112.08679

Key idea: augment graph embeddings by adding uniform noise ε at each layer,
then use InfoNCE contrastive loss between the two noisy views.

Loss = BPR + λ · InfoNCE(view1, view2)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel


class SimGCL(BaseModel):
    """
    SimGCL: Simple Graph Contrastive Learning for Recommendation.

    Args:
        n_users:       Number of users.
        n_items:       Number of items.
        embedding_dim: Embedding dimension.
        n_layers:      LightGCN layers.
        eps:           Noise magnitude for augmentation.
        temperature:   InfoNCE temperature τ.
        lambda_cl:     Contrastive loss weight λ.
        norm_adj:      Normalised adjacency sparse tensor.
        device:        Torch device.
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
        norm_adj: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(n_users, n_items, embedding_dim, device)
        self.n_layers = n_layers
        self.eps = eps
        self.temperature = temperature
        self.lambda_cl = lambda_cl

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        if norm_adj is not None:
            self.register_buffer("norm_adj", norm_adj)
        else:
            self.norm_adj = None

        self._init_weights()

    # ── Graph propagation (with or without noise) ─────────────────────────────

    def _propagate(
        self, perturb: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LightGCN-style propagation with optional per-layer noise injection.

        Args:
            perturb: If True, add uniform noise ε at each layer (augmented view).

        Returns:
            user_emb [n_users, D], item_emb [n_items, D]
        """
        # Lazy device move: adj follows model device (CPU/GPU transparent)
        _dev = self.user_embedding.weight.device
        adj = self.norm_adj.to(_dev)
        E0 = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        # Running mean — avoids storing K+1 tensors on GPU simultaneously
        E_k = E0
        acc = E0.clone()

        for _ in range(self.n_layers):
            E_k = torch.sparse.mm(adj, E_k)
            if perturb:
                noise = torch.rand_like(E_k) * 2 - 1
                noise = F.normalize(noise, dim=-1) * self.eps
                E_k = E_k + noise
            acc = acc + E_k

        E_final = acc / (self.n_layers + 1)
        return E_final[: self.n_users], E_final[self.n_users :]

    # ── BaseModel interface ───────────────────────────────────────────────────

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns BPR embeddings + two augmented user views for CL loss.

        Optimization: clean propagation is run once and reused for BPR embeddings.
        Only the two perturbed views require additional propagation.

        Returns:
            user_emb, pos_emb, neg_emb:  for BPR
            user_view1, user_view2:       two augmented user embeddings [B, D]
        """
        # Clean propagation (run once, shared for BPR)
        user_emb, item_emb = self._propagate(perturb=False)
        u_emb = user_emb[users]
        p_emb = item_emb[pos_items]
        n_emb = item_emb[neg_items]

        # Two augmented views (perturbed) — separate forward passes needed for CL
        u1, _ = self._propagate(perturb=True)
        u2, _ = self._propagate(perturb=True)
        return u_emb, p_emb, n_emb, u1[users], u2[users]

    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return clean (non-perturbed) embeddings for evaluation."""
        return self._propagate(perturb=False)

    def contrastive_loss(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss between two views of the same users.

        Args:
            view1: [B, D]
            view2: [B, D]

        Returns:
            Scalar contrastive loss.
        """
        v1 = F.normalize(view1, dim=-1)
        v2 = F.normalize(view2, dim=-1)

        # Cosine similarity matrix [B, B]
        sim = torch.matmul(v1, v2.T) / self.temperature

        # Diagonal = positive pairs
        labels = torch.arange(len(v1), device=v1.device)
        loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)
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

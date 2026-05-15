# _lazy_device_fixed_
"""
models/lightgcn.py
─────────────────────────────────────────────────────────────────────────────
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
He et al., SIGIR 2020 — https://arxiv.org/abs/2002.02126

Graph propagation (K layers, mean pooling):
    E^(k+1) = Â · E^(k)
    E_final  = (1/K+1) Σ_{k=0}^{K} E^(k)

where Â = D^{-1/2} A D^{-1/2} is the symmetrically normalised adjacency
of the user-item bipartite graph.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel


class LightGCN(BaseModel):
    """
    LightGCN recommender model.

    Args:
        n_users:       Number of users.
        n_items:       Number of items.
        embedding_dim: Embedding dimension (fairness: 64).
        n_layers:      Number of graph convolution layers K.
        norm_adj:      Pre-built normalised adjacency sparse tensor [N+M, N+M].
        device:        Torch device.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        norm_adj: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(n_users, n_items, embedding_dim, device)
        self.n_layers = n_layers

        # Learnable initial embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Normalised adjacency (not a parameter — moved to device externally)
        if norm_adj is not None:
            self.register_buffer("norm_adj", norm_adj)
        else:
            self.norm_adj = None

        self._init_weights()

    # ── Graph propagation ─────────────────────────────────────────────────────

    def _graph_propagation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run K-layer LightGCN propagation and return mean-pooled embeddings.

        Returns:
            user_final: [n_users, D]
            item_final: [n_items, D]
        """
        if self.norm_adj is None:
            raise RuntimeError("norm_adj not set. Pass it at construction or via set_adj().")

        # Concatenate user + item embeddings → [N+M, D]
        E0 = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        # Accumulate layer outputs for mean pooling
        # Running mean accumulation — avoids storing K+1 full tensors on GPU
        # Memory: O(2 × [N+M, D]) instead of O((K+1) × [N+M, D])
        # Lazy device move: adj follows model device (CPU/GPU transparent)
        _dev = self.user_embedding.weight.device
        adj = self.norm_adj.to(_dev)
        E_k = E0
        acc = E0.clone()  # accumulator for mean pooling (layer 0 included)

        for _ in range(self.n_layers):
            E_k = torch.sparse.mm(adj, E_k)
            acc = acc + E_k

        E_final = acc / (self.n_layers + 1)  # mean over K+1 layers

        user_final = E_final[: self.n_users]
        item_final = E_final[self.n_users :]
        return user_final, item_final

    # ── BaseModel interface ───────────────────────────────────────────────────

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass.

        Args:
            users:     [B] user IDs.
            pos_items: [B] positive item IDs.
            neg_items: [B] negative item IDs.

        Returns:
            user_emb:  [B, D] propagated user embeddings.
            pos_emb:   [B, D] propagated positive item embeddings.
            neg_emb:   [B, D] propagated negative item embeddings.
        """
        user_final, item_final = self._graph_propagation()

        user_emb = user_final[users]
        pos_emb = item_final[pos_items]
        neg_emb = item_final[neg_items]
        return user_emb, pos_emb, neg_emb

    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return full propagated user and item embedding matrices."""
        return self._graph_propagation()

    # ── L2 regularisation (on initial embeddings only, as per paper) ─────────

    def l2_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """L2 regularisation loss on E^0 (initial embeddings)."""
        u0 = self.user_embedding(users)
        p0 = self.item_embedding(pos_items)
        n0 = self.item_embedding(neg_items)
        return (u0.norm(2).pow(2) + p0.norm(2).pow(2) + n0.norm(2).pow(2)) / (2 * len(users))

    def set_adj(self, norm_adj: torch.Tensor) -> None:
        """Set or update the normalised adjacency matrix."""
        self.register_buffer("norm_adj", norm_adj)

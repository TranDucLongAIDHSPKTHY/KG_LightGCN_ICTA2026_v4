"""
models/base_model.py
─────────────────────────────────────────────────────────────────────────────
Abstract base class for all recommender models.
Enforces shared interface: embedding_dim, forward, predict, get_embeddings.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    Abstract recommender model.

    All models share:
      - embedding_dim = 64  (HARD fairness constraint)
      - A unified predict() method for full-ranking evaluation
      - Weight initialisation via _init_weights()
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            n_users:       Number of users.
            n_items:       Number of items.
            embedding_dim: Embedding dimension (default 64, fairness constraint).
            device:        Torch device.
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.device = device or torch.device("cpu")

    # ── Interface ─────────────────────────────────────────────────────────────

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Training forward pass.
        Should return embeddings or loss components consumed by a loss function.
        """

    @abstractmethod
    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return final user and item embedding matrices after graph propagation.

        Returns:
            user_emb: [n_users, embedding_dim]
            item_emb: [n_items, embedding_dim]
        """

    def predict(
        self,
        users: torch.Tensor,
        items: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score users against a set of items (or all items).

        Args:
            users:  1-D LongTensor of user IDs [B].
            items:  Optional 1-D or 2-D LongTensor [I] or [B, I].
                    If None, scores against ALL items.

        Returns:
            scores: [B, I] float tensor.
        """
        user_emb, item_emb = self.get_embeddings()

        # Gather user embeddings
        u = user_emb[users]  # [B, D]

        if items is None:
            # Full ranking: dot with all items
            scores = torch.matmul(u, item_emb.T)  # [B, n_items]
        elif items.dim() == 1:
            # Same item set for all users
            i = item_emb[items]  # [I, D]
            scores = torch.matmul(u, i.T)  # [B, I]
        else:
            # Per-user item sets: items [B, I]
            i = item_emb[items]  # [B, I, D]
            scores = torch.bmm(u.unsqueeze(1), i.transpose(1, 2)).squeeze(1)  # [B, I]

        return scores

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for all embedding layers."""
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_users={self.n_users}, n_items={self.n_items}, "
            f"emb_dim={self.embedding_dim}, params={self.parameter_count():,})"
        )

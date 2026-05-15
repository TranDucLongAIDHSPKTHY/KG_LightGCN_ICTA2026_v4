# _lazy_device_fixed_
"""
models/kgcl.py
─────────────────────────────────────────────────────────────────────────────
KGCL: Knowledge Graph Contrastive Learning for Recommendation
Yang et al., SIGIR 2022 — https://arxiv.org/abs/2205.00976

Key contributions:
  1. KG-guided user-item graph augmentation (drop items not linked to KG entities)
  2. Cross-view contrastive loss between KG-augmented and original CF views
  3. Joint training: BPR + KG embedding loss + CL loss

Augmentation strategy:
  - For each user's interactions: keep item i with probability p(i)
    where p(i) ∝ degree of item's KG entity
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel


class KGCL(BaseModel):
    """
    Knowledge Graph Contrastive Learning for Recommendation.

    Args:
        n_users:       Number of users.
        n_items:       Number of items.
        n_entities:    Total KG entity count.
        n_relations:   Number of KG relation types.
        embedding_dim: Embedding dimension.
        n_layers:      CF propagation layers.
        kg_n_layers:   KG propagation layers.
        temp:          Contrastive loss temperature.
        lambda_kg:     KG loss weight.
        norm_adj:      Normalised CF adjacency.
        kg_triples:    np.ndarray [T, 3] of KG triples.
        device:        Torch device.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_entities: int,
        n_relations: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        kg_n_layers: int = 2,
        temp: float = 0.2,
        lambda_kg: float = 0.5,
        norm_adj: Optional[torch.Tensor] = None,
        kg_triples: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(n_users, n_items, embedding_dim, device)
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.n_layers = n_layers
        self.kg_n_layers = kg_n_layers
        self.temp = temp
        self.lambda_kg = lambda_kg

        # CF embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.entity_embedding = nn.Embedding(n_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(n_relations, embedding_dim)

        # Relation gate (KGCL uses element-wise gating)
        self.relation_gate = nn.Linear(embedding_dim, embedding_dim, bias=False)

        if norm_adj is not None:
            self.register_buffer("norm_adj", norm_adj)
        else:
            self.norm_adj = None

        # KG entity degree (used for augmentation probability)
        self.item_kg_degree: Optional[torch.Tensor] = None
        if kg_triples is not None:
            self._build_kg_degree(kg_triples, n_entities)

        self._init_weights()

    # ── KG degree ─────────────────────────────────────────────────────────────

    def _build_kg_degree(
        self, kg_triples: np.ndarray, n_entities: int
    ) -> None:
        """Compute entity degree in KG (used for augmentation probability)."""
        degree = np.zeros(n_entities, dtype=np.float32)
        for h, _, t in kg_triples:
            degree[h] += 1
            degree[t] += 1
        # Normalise to [0, 1]
        max_deg = degree.max()
        if max_deg > 0:
            degree /= max_deg
        # Item degree (item_id == entity_id assumed)
        item_degree = degree[: self.n_items]
        self.item_kg_degree = torch.tensor(item_degree, dtype=torch.float32)

    # ── KG entity propagation ─────────────────────────────────────────────────

    def _kg_propagation(self) -> torch.Tensor:
        """
        Light-weight relational message passing over KG.
        Returns enriched entity embeddings [n_entities, D].
        """
        if self.norm_adj is None:
            return self.entity_embedding.weight

        E = self.entity_embedding.weight
        # Simple mean aggregation (no per-relation attention for efficiency)
        # A full KGCL implementation would use relation-specific graphs;
        # here we use a single entity-entity adjacency (set externally).
        if hasattr(self, "kg_norm_adj") and self.kg_norm_adj is not None:
            _dev = self.user_embedding.weight.device
            adj = self.kg_norm_adj.to(_dev)
            # Running mean — no list of full tensors
            E_k = E
            acc = E.clone()
            for _ in range(self.kg_n_layers):
                E_k = torch.sparse.mm(adj, E_k)
                acc = acc + E_k
            E = acc / (self.kg_n_layers + 1)
        return E

    # ── CF propagation ────────────────────────────────────────────────────────

    def _cf_propagation(
        self, adj: torch.Tensor, entity_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LightGCN propagation using given adjacency and entity embeddings for items.
        """
        item_e = entity_emb[: self.n_items]
        E0 = torch.cat([self.user_embedding.weight, item_e], dim=0)
        E_k = E0
        acc = E0.clone()
        for _ in range(self.n_layers):
            E_k = torch.sparse.mm(adj, E_k)
            acc = acc + E_k
        E_final = acc / (self.n_layers + 1)
        return E_final[: self.n_users], E_final[self.n_users :]

    # ── KG-guided augmentation ────────────────────────────────────────────────

    def _augment_adj(
        self, adj: torch.Tensor, drop_prob: float = 0.1
    ) -> torch.Tensor:
        """
        Vectorized KG-guided edge dropout.
        Items with lower KG connectivity have higher drop probability.
        Replaces the slow per-edge Python loop with full-batch tensor ops.
        """
        if self.item_kg_degree is None:
            return self._random_edge_drop(adj, drop_prob)

        adj_coo = adj.coalesce()
        indices = adj_coo.indices()   # [2, nnz]
        values  = adj_coo.values()    # [nnz]
        n_u = self.n_users
        device = adj.device

        item_keep = self.item_kg_degree.to(device)  # [n_items], in [0,1]

        rows, cols = indices[0], indices[1]

        # Determine item node index per edge (vectorized)
        # Edges where row >= n_users: item is on the left side
        row_is_item = rows >= n_u
        col_is_item = cols >= n_u
        has_item = row_is_item | col_is_item

        # Compute item node index per edge (0-indexed)
        item_node = torch.where(row_is_item, rows - n_u, cols - n_u)
        item_node = item_node.clamp(0, len(item_keep) - 1)

        # Keep probability: higher KG degree → lower drop prob
        p_keep = item_keep[item_node]  # [nnz]
        # For non-item edges, always keep
        p_keep = torch.where(has_item, p_keep + (1.0 - drop_prob), torch.ones_like(p_keep))
        p_keep = p_keep.clamp(0.0, 1.0)

        keep_mask = torch.rand_like(p_keep) < p_keep
        new_indices = indices[:, keep_mask]
        new_values  = values[keep_mask]
        return torch.sparse_coo_tensor(
            new_indices, new_values, adj.shape, device=device
        ).coalesce()

    @staticmethod
    def _random_edge_drop(adj: torch.Tensor, drop_prob: float) -> torch.Tensor:
        """Randomly drop edges with probability drop_prob."""
        adj_coo = adj.coalesce()
        values = adj_coo.values()
        mask = torch.rand_like(values) > drop_prob
        return torch.sparse_coo_tensor(
            adj_coo.indices()[:, mask],
            values[mask],
            adj.shape,
            device=adj.device,
        )

    # ── BaseModel interface ───────────────────────────────────────────────────

    def refresh_augmented_views(self) -> None:
        """
        Pre-compute two augmented adjacency matrices for this epoch.
        Call once per epoch from the trainer to avoid rebuilding per-batch.
        The augmented adj matrices are stored as instance vars and reused
        across all batches in the epoch.
        """
        # Lazy device move: adj follows model device (CPU/GPU transparent)
        _dev = self.user_embedding.weight.device
        adj = self.norm_adj.to(_dev)
        self._aug_adj1 = self._augment_adj(adj)
        self._aug_adj2 = self._augment_adj(adj)

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        """
        Returns:
            user_emb, pos_emb, neg_emb:  BPR embeddings
            user_view1, user_view2:       CL user embeddings [B, D]

        Uses epoch-level cached augmented views (set by refresh_augmented_views()).
        Fallback: generates new views per call if cache not set.
        """
        # Lazy device move: adj follows model device (CPU/GPU transparent)
        _dev = self.user_embedding.weight.device
        adj = self.norm_adj.to(_dev)
        entity_emb = self._kg_propagation()

        user_emb, item_emb = self._cf_propagation(adj, entity_emb)

        # Use epoch-cached augmented adjs to avoid rebuilding per-batch
        _aug1 = getattr(self, '_aug_adj1', None)
        _aug2 = getattr(self, '_aug_adj2', None)
        adj1 = _aug1 if _aug1 is not None else self._augment_adj(adj)
        adj2 = _aug2 if _aug2 is not None else self._augment_adj(adj)
        u1, _ = self._cf_propagation(adj1, entity_emb)
        u2, _ = self._cf_propagation(adj2, entity_emb)

        return (
            user_emb[users], item_emb[pos_items], item_emb[neg_items],
            u1[users], u2[users],
        )

    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            entity_emb = self._kg_propagation()
            return self._cf_propagation(adj, entity_emb)

    def contrastive_loss(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> torch.Tensor:
        v1 = F.normalize(view1, dim=-1)
        v2 = F.normalize(view2, dim=-1)
        sim = torch.matmul(v1, v2.T) / self.temp
        labels = torch.arange(len(v1), device=v1.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss

    def l2_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        u0 = self.user_embedding(users)
        p0 = self.entity_embedding(pos_items)
        n0 = self.entity_embedding(neg_items)
        return (u0.norm(2).pow(2) + p0.norm(2).pow(2) + n0.norm(2).pow(2)) / (2 * len(users))

    def set_adj(self, norm_adj: torch.Tensor) -> None:
        self.register_buffer("norm_adj", norm_adj)

    def set_kg_norm_adj(self, kg_norm_adj: torch.Tensor) -> None:
        self.register_buffer("kg_norm_adj", kg_norm_adj)

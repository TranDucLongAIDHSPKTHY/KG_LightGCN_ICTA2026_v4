"""
models/kgcl.py
─────────────────────────────────────────────────────────────────────────────
KGCL: Knowledge Graph Contrastive Learning for Recommendation
Yang et al., SIGIR 2022 — https://arxiv.org/abs/2205.00976

Fixes vs v4:
  [BUG-1] get_embeddings(): `adj` was undefined (NameError).
          Fixed: use self.norm_adj inside get_embeddings.
  [BUG-2] _kg_propagation(): relation embeddings were completely ignored.
          KGCL (Eq.5): m_{h←t,r} = e_t ⊙ e_r  (element-wise gate by relation).
          Fixed: incorporate relation embeddings via set_kg_edges().
  [BUG-3] _augment_adj(): keep-probability formula was wrong.
          Old: p_keep = item_keep + (1 - drop_prob)  → always > 1, no dropout.
          Fixed: p_keep = drop_prob + (1 - drop_prob)*degree  ∈ [drop_prob, 1].
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel


class KGCL(BaseModel):
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

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.entity_embedding = nn.Embedding(n_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(n_relations, embedding_dim)

        if norm_adj is not None:
            self.register_buffer("norm_adj", norm_adj)
        else:
            self.norm_adj = None

        self.item_kg_degree: Optional[torch.Tensor] = None
        if kg_triples is not None:
            self._build_kg_degree(kg_triples, n_entities)

        # Per-edge KG tensors for relation-aware propagation
        self._kg_heads: Optional[torch.Tensor] = None
        self._kg_tails: Optional[torch.Tensor] = None
        self._kg_rels: Optional[torch.Tensor] = None

        self._init_weights()

    # ── KG degree ─────────────────────────────────────────────────────────────

    def _build_kg_degree(self, kg_triples: np.ndarray, n_entities: int) -> None:
        degree = np.zeros(n_entities, dtype=np.float32)
        for h, _, t in kg_triples:
            degree[h] += 1
            degree[t] += 1
        max_deg = degree.max()
        if max_deg > 0:
            degree /= max_deg
        self.item_kg_degree = torch.tensor(degree[: self.n_items], dtype=torch.float32)

    # ── KG entity propagation ─────────────────────────────────────────────────

    def _kg_propagation(self) -> torch.Tensor:
        """
        Relation-aware message passing (KGCL Eq.5):
            m_{h←(t,r)} = e_t ⊙ e_r
            e_h' = mean_aggregation(m) with residual + L2-norm
        Falls back to simple mean via kg_norm_adj if edge tensors not set.
        """
        E = self.entity_embedding.weight  # [n_entities, D]

        # FIX [BUG-2]: relation-aware path
        if (self._kg_heads is not None
                and self._kg_tails is not None
                and self._kg_rels is not None):
            device = E.device
            heads = self._kg_heads.to(device)
            tails = self._kg_tails.to(device)
            rels  = self._kg_rels.to(device)

            E_k = E
            for _ in range(self.kg_n_layers):
                t_emb = E_k[tails]                              # [E_kg, D]
                r_emb = self.relation_embedding(rels)           # [E_kg, D]
                msgs  = t_emb * r_emb                          # element-wise gate

                agg   = torch.zeros_like(E_k)
                count = torch.zeros(self.n_entities, device=device)
                agg.scatter_add_(0, heads.unsqueeze(1).expand_as(msgs), msgs)
                count.scatter_add_(0, heads, torch.ones(len(heads), device=device))
                count = count.clamp(min=1).unsqueeze(1)
                E_k = F.normalize(E_k + agg / count, dim=-1)

            return E_k

        # Fallback: simple LightGCN-style mean over kg_norm_adj
        if hasattr(self, "kg_norm_adj") and self.kg_norm_adj is not None:
            _dev = self.user_embedding.weight.device
            adj  = self.kg_norm_adj.to(_dev)
            E_k  = E
            acc  = E.clone()
            for _ in range(self.kg_n_layers):
                E_k = torch.sparse.mm(adj, E_k)
                acc = acc + E_k
            return acc / (self.kg_n_layers + 1)

        return E

    # ── CF propagation ────────────────────────────────────────────────────────

    def _cf_propagation(
        self, adj: torch.Tensor, entity_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _augment_adj(self, adj: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
        """
        KG-guided edge dropout.
        FIX [BUG-3]: p_keep = drop_prob + (1 - drop_prob) * degree ∈ [drop_prob, 1].
        """
        if self.item_kg_degree is None:
            return self._random_edge_drop(adj, drop_prob)

        adj_coo  = adj.coalesce()
        indices  = adj_coo.indices()
        values   = adj_coo.values()
        device   = adj.device
        n_u      = self.n_users

        item_keep = self.item_kg_degree.to(device)
        rows, cols = indices[0], indices[1]

        row_is_item = rows >= n_u
        col_is_item = cols >= n_u
        has_item    = row_is_item | col_is_item
        item_node   = torch.where(row_is_item, rows - n_u, cols - n_u)
        item_node   = item_node.clamp(0, len(item_keep) - 1)

        # FIX: correct keep probability formula
        p_keep = drop_prob + (1.0 - drop_prob) * item_keep[item_node]
        p_keep = torch.where(has_item, p_keep, torch.ones_like(p_keep))
        p_keep = p_keep.clamp(0.0, 1.0)

        keep_mask   = torch.rand_like(p_keep) < p_keep
        new_indices = indices[:, keep_mask]
        new_values  = values[keep_mask]
        return torch.sparse_coo_tensor(
            new_indices, new_values, adj.shape, device=device
        ).coalesce()

    @staticmethod
    def _random_edge_drop(adj: torch.Tensor, drop_prob: float) -> torch.Tensor:
        adj_coo = adj.coalesce()
        values  = adj_coo.values()
        mask    = torch.rand_like(values) > drop_prob
        return torch.sparse_coo_tensor(
            adj_coo.indices()[:, mask],
            values[mask],
            adj.shape,
            device=adj.device,
        )

    # ── BaseModel interface ───────────────────────────────────────────────────

    def refresh_augmented_views(self) -> None:
        """Pre-compute two augmented adj matrices once per epoch."""
        _dev = self.user_embedding.weight.device
        adj  = self.norm_adj.to(_dev)
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
        _dev = self.user_embedding.weight.device
        adj  = self.norm_adj.to(_dev)
        entity_emb = self._kg_propagation()

        user_emb, item_emb = self._cf_propagation(adj, entity_emb)

        _aug1 = getattr(self, '_aug_adj1', None)
        _aug2 = getattr(self, '_aug_adj2', None)
        adj1  = _aug1 if _aug1 is not None else self._augment_adj(adj)
        adj2  = _aug2 if _aug2 is not None else self._augment_adj(adj)
        u1, _ = self._cf_propagation(adj1, entity_emb)
        u2, _ = self._cf_propagation(adj2, entity_emb)

        return (
            user_emb[users], item_emb[pos_items], item_emb[neg_items],
            u1[users], u2[users],
        )

    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """FIX [BUG-1]: `adj` was undefined. Now correctly uses self.norm_adj."""
        with torch.no_grad():
            _dev = self.user_embedding.weight.device
            adj  = self.norm_adj.to(_dev)          # FIX
            entity_emb = self._kg_propagation()
            return self._cf_propagation(adj, entity_emb)

    def contrastive_loss(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> torch.Tensor:
        v1  = F.normalize(view1, dim=-1)
        v2  = F.normalize(view2, dim=-1)
        sim = torch.matmul(v1, v2.T) / self.temp
        labels = torch.arange(len(v1), device=v1.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

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

    def set_kg_edges(
        self,
        heads: torch.Tensor,
        tails: torch.Tensor,
        rels: torch.Tensor,
    ) -> None:
        """
        Set per-edge KG tensors for relation-aware propagation.
        Args:
            heads: [E_kg] head entity indices (long).
            tails: [E_kg] tail entity indices (long).
            rels:  [E_kg] relation indices (long).
        """
        self._kg_heads = heads
        self._kg_tails = tails
        self._kg_rels  = rels
# _lazy_device_fixed_
"""
models/kgat.py
─────────────────────────────────────────────────────────────────────────────
KGAT: Knowledge Graph Attention Network for Recommendation
Wang et al., KDD 2019 — https://arxiv.org/abs/1905.07854

Architecture:
  1. TransR-based KG embedding loss (entity + relation embeddings)
  2. Attentive aggregation over KG neighbourhoods → entity embeddings
  3. Graph-based CF propagation on user-item graph (using entity embeddings for items)
  4. Bi-interaction / GCN aggregation

This implementation follows the original paper with bi-interaction aggregation.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel


class KGAT(BaseModel):
    """
    Knowledge Graph Attention Network.

    Args:
        n_users:       Number of users.
        n_items:       Number of items.
        n_entities:    Total number of KG entities (items are subset of entities).
        n_relations:   Number of KG relation types.
        embedding_dim: Embedding dimension (fairness: 64).
        relation_dim:  Relation embedding dimension.
        n_layers:      CF propagation layers.
        agg_type:      Aggregation type: 'bi-interaction' | 'gcn' | 'graphsage'.
        norm_adj:      User-item normalised adjacency [N+M, N+M].
        device:        Torch device.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_entities: int,
        n_relations: int,
        embedding_dim: int = 64,
        relation_dim: int = 64,
        n_layers: int = 3,
        agg_type: str = "bi-interaction",
        norm_adj: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(n_users, n_items, embedding_dim, device)
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.relation_dim = relation_dim
        self.n_layers = n_layers
        self.agg_type = agg_type

        # CF embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        # Entity embeddings (items are entities; item_id maps to entity_id)
        self.entity_embedding = nn.Embedding(n_entities, embedding_dim)
        # Relation embeddings (for TransR)
        self.relation_embedding = nn.Embedding(n_relations, relation_dim)
        # Projection matrices W_r (per relation) for TransR
        self.trans_w = nn.Embedding(n_relations, embedding_dim * relation_dim)

        # Layer-wise transformation matrices for BI aggregation
        if agg_type == "bi-interaction":
            self.W_gc = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim, bias=False)
                for _ in range(n_layers)
            ])
            self.W_bi = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim, bias=False)
                for _ in range(n_layers)
            ])
        else:
            # GCN / GraphSAGE
            self.W_gc = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim, bias=False)
                for _ in range(n_layers)
            ])

        # Attention: score(head, relation, tail)
        self.attn_W = nn.Linear(embedding_dim, 1, bias=False)

        if norm_adj is not None:
            self.register_buffer("norm_adj", norm_adj)
        else:
            self.norm_adj = None

        # Adjacency list for KG (set via set_kg_adj)
        self.kg_adj: Dict[int, List[Tuple[int, int]]] = {}
        self._kg_tensors = None  # cached edge tensors (built on first forward pass)

        self._init_weights()
        nn.init.xavier_uniform_(self.trans_w.weight)

    # ── KG embedding (TransR) ─────────────────────────────────────────────────

    def _project(
        self,
        entity_emb: torch.Tensor,
        relation_id: torch.Tensor,
    ) -> torch.Tensor:
        """Project entity embedding into relation space via W_r."""
        W = self.trans_w(relation_id)  # [B, D*Rd]
        W = W.view(-1, self.embedding_dim, self.relation_dim)  # [B, D, Rd]
        # entity_emb: [B, D] → [B, 1, D]
        proj = torch.bmm(entity_emb.unsqueeze(1), W).squeeze(1)  # [B, Rd]
        return proj

    def kg_forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_tails: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TransR scoring for KG triples.

        Returns:
            pos_scores: [B]  ||h + r - t_pos||
            neg_scores: [B]  ||h + r - t_neg||
        """
        h = self.entity_embedding(heads)
        r = self.relation_embedding(relations)
        t_pos = self.entity_embedding(pos_tails)
        t_neg = self.entity_embedding(neg_tails)

        h_proj = self._project(h, relations)
        tp_proj = self._project(t_pos, relations)
        tn_proj = self._project(t_neg, relations)

        pos_score = torch.sum((h_proj + r - tp_proj) ** 2, dim=-1)
        neg_score = torch.sum((h_proj + r - tn_proj) ** 2, dim=-1)
        return pos_score, neg_score

    # ── Attentive KG aggregation ──────────────────────────────────────────────

    def _build_kg_sparse_tensors(self, device: torch.device):
        """
        Build vectorized head/tail/relation tensors from kg_adj for fast aggregation.
        Called once and cached in self._kg_tensors.
        """
        if not self.kg_adj:
            return None
        heads_list, tails_list, rels_list = [], [], []
        for h, neighbours in self.kg_adj.items():
            for t, r in neighbours:
                heads_list.append(h)
                tails_list.append(t)
                rels_list.append(r)
        if not heads_list:
            return None
        heads_t = torch.tensor(heads_list, dtype=torch.long, device=device)
        tails_t = torch.tensor(tails_list, dtype=torch.long, device=device)
        rels_t  = torch.tensor(rels_list,  dtype=torch.long, device=device)
        return heads_t, tails_t, rels_t

    def _compute_entity_embeddings(self) -> torch.Tensor:
        """
        Vectorized attention-based KG aggregation to enrich entity embeddings.
        Replaces the slow per-node Python loop with full-batch tensor ops.
        """
        if not self.kg_adj:
            return self.entity_embedding.weight

        E = self.entity_embedding.weight  # [n_entities, D]
        device = E.device

        # Build/cache KG edge tensors (avoids rebuilding every epoch)
        if not hasattr(self, "_kg_tensors") or self._kg_tensors is None:
            self._kg_tensors = self._build_kg_sparse_tensors(device)
        if self._kg_tensors is None:
            return E

        heads_t, tails_t, rels_t = self._kg_tensors
        heads_t = heads_t.to(device)
        tails_t = tails_t.to(device)

        h_emb = E[heads_t]   # [E_total, D]
        t_emb = E[tails_t]   # [E_total, D]

        # Attention score (element-wise product → linear)
        attn_raw = self.attn_W(h_emb * t_emb).squeeze(-1)  # [E_total]

        # Softmax per head (scatter softmax)
        # Use scatter for normalisation across each head's neighbours
        n_edges = heads_t.shape[0]
        attn_exp = torch.exp(attn_raw - attn_raw.max())  # stable softmax
        # Sum per head
        attn_sum = torch.zeros(self.n_entities, device=device)
        attn_sum.scatter_add_(0, heads_t, attn_exp)
        attn_norm = attn_exp / (attn_sum[heads_t] + 1e-8)  # [E_total]

        # Weighted sum of tail embeddings per head
        aggregated = torch.zeros_like(E)
        aggregated.scatter_add_(0, heads_t.unsqueeze(1).expand_as(t_emb),
                                attn_norm.unsqueeze(1) * t_emb)

        # Residual connection
        return E + aggregated

    # ── CF propagation ────────────────────────────────────────────────────────

    def _cf_propagation(
        self, entity_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Graph CF propagation using enriched entity embeddings for items.

        Returns:
            user_final [n_users, D], item_final [n_items, D]
        """
        if self.norm_adj is None:
            raise RuntimeError("norm_adj not set.")

        # Use entity embeddings for items (item_id == entity_id in KGCL convention)
        item_e = entity_emb[: self.n_items]
        E0 = torch.cat([self.user_embedding.weight, item_e], dim=0)

        # Lazy device move: adj follows model device (CPU/GPU transparent)
        _dev = self.user_embedding.weight.device
        adj = self.norm_adj.to(_dev)
        # Running mean — avoids storing K+1 full tensors on GPU
        E_k = E0
        acc = E0.clone()

        for l in range(self.n_layers):
            nb = torch.sparse.mm(adj, E_k)
            if self.agg_type == "bi-interaction":
                E_k = F.leaky_relu(self.W_gc[l](E_k + nb)) + F.leaky_relu(
                    self.W_bi[l](E_k * nb)
                )
            elif self.agg_type == "graphsage":
                E_k = F.leaky_relu(self.W_gc[l](torch.cat([E_k, nb], dim=-1)))
            else:  # gcn
                E_k = F.leaky_relu(self.W_gc[l](nb))
            acc = acc + E_k

        E_final = acc / (self.n_layers + 1)
        return E_final[: self.n_users], E_final[self.n_users :]

    # ── BaseModel interface ───────────────────────────────────────────────────

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        precomputed_entity_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Use precomputed entity embeddings if provided (set by KGTrainer each epoch).
        # This avoids re-running the expensive KG aggregation for every batch.
        entity_emb = precomputed_entity_emb if precomputed_entity_emb is not None             else self._compute_entity_embeddings()
        user_final, item_final = self._cf_propagation(entity_emb)
        return user_final[users], item_final[pos_items], item_final[neg_items]

    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            entity_emb = self._compute_entity_embeddings()
            return self._cf_propagation(entity_emb)

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

    def set_kg_adj(self, kg_adj: Dict[int, List[Tuple[int, int]]]) -> None:
        """Set KG adjacency list (head → [(tail, relation), …])."""
        self.kg_adj = kg_adj
        self._kg_tensors = None  # invalidate cache

    def to(self, *args, **kwargs):
        """Override to() to invalidate KG tensor cache on device change."""
        self._kg_tensors = None
        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        """Override cuda() to invalidate KG tensor cache."""
        self._kg_tensors = None
        return super().cuda(device)

    def cpu(self):
        """Override cpu() to invalidate KG tensor cache."""
        self._kg_tensors = None
        return super().cpu()

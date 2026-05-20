"""
models/kgat.py
─────────────────────────────────────────────────────────────────────────────
KGAT: Knowledge Graph Attention Network for Recommendation
Wang et al., KDD 2019 — https://arxiv.org/abs/1905.07854

Architecture:
  1. TransR-based KG embedding loss (entity + relation embeddings)
  2. Attentive aggregation over KG neighbourhoods → entity embeddings
     (multi-hop: repeated kg_n_layers times)
  3. Graph-based CF propagation using enriched entity embeddings for items
  4. Bi-interaction / GCN / GraphSAGE aggregation

Fixes vs previous version:
  [BUG-K3-FIX] CF propagation: W_out input dim phải là (n_layers+1)*D vì
               KGAT concat E^0 || E^1 || ... || E^K (bao gồm cả E^0).
               Version cũ dùng n_layers*D → bỏ mất layer-0 embedding.

  [BUG-K4-FIX] _compute_entity_embeddings(): attention aggregation phải
               lặp kg_n_layers lần (multi-hop). Version cũ chỉ chạy 1 lần.
               Paper (Eq.6): e^(l+1) = LeakyReLU(W_l · Aggregate(e^l))

  [OOM-FIX] Giữ nguyên chunked projection để không OOM trên 4 GB VRAM.
"""

import gc
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel

_DEFAULT_CHUNK = 32_768


class KGAT(BaseModel):
    """
    Knowledge Graph Attention Network.

    Args:
        n_users:       Number of users.
        n_items:       Number of items.
        n_entities:    Total number of KG entities.
        n_relations:   Number of KG relation types.
        embedding_dim: Embedding dimension (fairness: 64).
        relation_dim:  Relation embedding dimension (= embedding_dim in paper).
        n_layers:      CF propagation layers.
        agg_type:      'bi-interaction' | 'gcn' | 'graphsage'.
        norm_adj:      User-item normalised adjacency [N+M, N+M].
        device:        Torch device.
        chunk_size:    Triples processed per chunk in _compute_entity_embeddings.
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
        kg_n_layers: int = 2,
        agg_type: str = "bi-interaction",
        norm_adj: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        chunk_size: int = _DEFAULT_CHUNK,
    ) -> None:
        super().__init__(n_users, n_items, embedding_dim, device)
        self.n_entities   = n_entities
        self.n_relations  = n_relations
        self.relation_dim = relation_dim
        self.n_layers     = n_layers
        self.kg_n_layers  = kg_n_layers
        self.agg_type     = agg_type
        self.chunk_size   = chunk_size

        # ── Embeddings ────────────────────────────────────────────────────────
        self.user_embedding     = nn.Embedding(n_users,    embedding_dim)
        self.entity_embedding   = nn.Embedding(n_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(n_relations, relation_dim)
        # TransR projection matrices W_r ∈ R^{D × Rd} per relation
        self.trans_w = nn.Embedding(n_relations, embedding_dim * relation_dim)

        # ── KG aggregation weights (per hop layer) ────────────────────────────
        # Multi-hop: kg_n_layers aggregation weight matrices
        self.W_kg = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim, bias=False)
            for _ in range(kg_n_layers)
        ])

        # ── CF aggregation weights ────────────────────────────────────────────
        if agg_type == "bi-interaction":
            self.W_gc = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim, bias=False)
                for _ in range(n_layers)
            ])
            self.W_bi = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim, bias=False)
                for _ in range(n_layers)
            ])
        elif agg_type == "graphsage":
            self.W_gc = nn.ModuleList([
                nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
                for _ in range(n_layers)
            ])
        else:  # gcn
            self.W_gc = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim, bias=False)
                for _ in range(n_layers)
            ])

        # [BUG-K3-FIX] W_out input dim = (n_layers+1)*D bao gồm E^0
        # Paper: e_u* = concat(e^0_u, e^1_u, ..., e^K_u)
        self.W_out = nn.Linear((n_layers + 1) * embedding_dim, embedding_dim, bias=False)

        # ── KG Attention ──────────────────────────────────────────────────────
        self.attn_V = nn.Linear(relation_dim, 1, bias=False)

        if norm_adj is not None:
            self.register_buffer("norm_adj", norm_adj)
        else:
            self.norm_adj = None

        self.kg_adj: Dict[int, List[Tuple[int, int]]] = {}
        self._kg_tensors_cpu: Optional[Tuple[torch.Tensor, ...]] = None

        self._init_weights()
        nn.init.xavier_uniform_(self.trans_w.weight)

    # ── KG embedding (TransR) ─────────────────────────────────────────────────

    def _project(
        self,
        entity_emb: torch.Tensor,   # [B, D]
        relation_id: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """Project entity embedding into relation space: e_proj = e · W_r ∈ R^Rd."""
        W = self.trans_w(relation_id)                          # [B, D*Rd]
        W = W.view(-1, self.embedding_dim, self.relation_dim)  # [B, D, Rd]
        proj = torch.bmm(entity_emb.unsqueeze(1), W).squeeze(1)  # [B, Rd]
        return proj

    def _project_chunked(
        self,
        entity_emb: torch.Tensor,
        relation_id: torch.Tensor,
        chunk_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """[OOM-FIX] Chunked version of _project."""
        out_chunks = []
        for start in range(0, entity_emb.size(0), chunk_size):
            end = min(start + chunk_size, entity_emb.size(0))
            e_chunk = entity_emb[start:end].to(device)
            r_chunk = relation_id[start:end].to(device)
            out_chunks.append(self._project(e_chunk, r_chunk).cpu())
            del e_chunk, r_chunk
        return torch.cat(out_chunks, dim=0)

    def kg_forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_tails: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TransR scoring: ||h_r + r - t_r||^2."""
        h   = self.entity_embedding(heads)
        r   = self.relation_embedding(relations)
        t_p = self.entity_embedding(pos_tails)
        t_n = self.entity_embedding(neg_tails)

        h_r  = self._project(h, relations)
        tp_r = self._project(t_p, relations)
        tn_r = self._project(t_n, relations)

        pos_score = torch.sum((h_r + r - tp_r) ** 2, dim=-1)
        neg_score = torch.sum((h_r + r - tn_r) ** 2, dim=-1)
        return pos_score, neg_score

    # ── Attentive KG aggregation (multi-hop) ──────────────────────────────────

    def _build_kg_sparse_tensors_cpu(self):
        """[OOM-FIX] Build head/tail/relation tensors on CPU."""
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
        return (
            torch.tensor(heads_list, dtype=torch.long),
            torch.tensor(tails_list, dtype=torch.long),
            torch.tensor(rels_list,  dtype=torch.long),
        )

    def _single_hop_aggregation(
        self,
        E: torch.Tensor,               # [n_entities, D] current entity emb (GPU)
        heads_cpu: torch.Tensor,
        tails_cpu: torch.Tensor,
        rels_cpu: torch.Tensor,
        hop: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        [BUG-K4-FIX] Một hop attention aggregation.

        KGAT paper (Eq.6): e^(l+1)_h = LeakyReLU(W_l · Σ_t π(h,r,t) · e^l_t)
        với π(h,r,t) = softmax_t[(W_r e_t)^T tanh(W_r e_h + e_r)]

        Returns: E_new [n_entities, D] (GPU)
        """
        E_total = heads_cpu.size(0)
        E_cpu   = E.detach().cpu()
        chunk   = self.chunk_size

        # Pass 1: tính attention exp scores
        attn_exp_cpu = torch.zeros(E_total)
        attn_sum_cpu = torch.zeros(self.n_entities)

        for start in range(0, E_total, chunk):
            end = min(start + chunk, E_total)
            h_idx      = heads_cpu[start:end]
            t_idx      = tails_cpu[start:end]
            r_idx      = rels_cpu[start:end]

            h_emb      = E_cpu[h_idx].to(device)
            t_emb      = E_cpu[t_idx].to(device)
            r_idx_gpu  = r_idx.to(device)
            r_emb      = self.relation_embedding(r_idx_gpu)

            h_proj = self._project(h_emb, r_idx_gpu)
            t_proj = self._project(t_emb, r_idx_gpu)

            gate     = torch.tanh(h_proj + r_emb)
            raw      = self.attn_V(t_proj * gate).squeeze(-1)
            exp_vals = torch.exp(raw - raw.max()).cpu()

            attn_exp_cpu[start:end] = exp_vals
            attn_sum_cpu.scatter_add_(0, h_idx, exp_vals)

            del h_emb, t_emb, r_emb, h_proj, t_proj, gate, raw, exp_vals, r_idx_gpu
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Pass 2: aggregate với normalised attention
        aggregated_cpu = torch.zeros_like(E_cpu)
        for start in range(0, E_total, chunk):
            end   = min(start + chunk, E_total)
            h_idx = heads_cpu[start:end]
            t_idx = tails_cpu[start:end]

            exp_vals  = attn_exp_cpu[start:end]
            denom     = attn_sum_cpu[h_idx] + 1e-8
            attn_norm = (exp_vals / denom).unsqueeze(1)

            t_emb_cpu = E_cpu[t_idx]
            weighted  = attn_norm * t_emb_cpu

            aggregated_cpu.scatter_add_(
                0,
                h_idx.unsqueeze(1).expand(-1, self.embedding_dim),
                weighted,
            )
            del weighted, t_emb_cpu, attn_norm, exp_vals, denom

        # LeakyReLU + W_kg projection + residual (Eq.6)
        aggregated_gpu = aggregated_cpu.to(device)
        E_new = F.leaky_relu(self.W_kg[hop](E + aggregated_gpu))
        E_new = F.normalize(E_new, dim=-1)

        del E_cpu, aggregated_cpu, aggregated_gpu
        gc.collect()

        return E_new

    def _compute_entity_embeddings(self) -> torch.Tensor:
        """
        [BUG-K4-FIX] Multi-hop attentive aggregation: lặp kg_n_layers lần.

        E^0 = entity_embedding.weight
        E^(l+1) = LeakyReLU(W_l · Aggregate_attn(E^l))
        Return: E^(kg_n_layers) [n_entities, D]
        """
        if not self.kg_adj:
            return self.entity_embedding.weight

        device = self.entity_embedding.weight.device

        if self._kg_tensors_cpu is None:
            self._kg_tensors_cpu = self._build_kg_sparse_tensors_cpu()
        if self._kg_tensors_cpu is None:
            return self.entity_embedding.weight

        heads_cpu, tails_cpu, rels_cpu = self._kg_tensors_cpu

        # [BUG-K4-FIX] Multi-hop: lặp kg_n_layers lần
        E = self.entity_embedding.weight  # [n_entities, D]
        for hop in range(self.kg_n_layers):
            E = self._single_hop_aggregation(
                E, heads_cpu, tails_cpu, rels_cpu, hop, device
            )

        return E  # [n_entities, D]

    # ── CF propagation ────────────────────────────────────────────────────────

    def _cf_propagation(
        self, entity_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Graph CF propagation.

        [BUG-K3-FIX] Concat [E^0, E^1, ..., E^K] (n_layers+1 tensors)
        rồi project qua W_out.
        """
        if self.norm_adj is None:
            raise RuntimeError("norm_adj not set.")

        item_e = entity_emb[: self.n_items]
        E0     = torch.cat([self.user_embedding.weight, item_e], dim=0)

        _dev = self.user_embedding.weight.device
        adj  = self.norm_adj.to(_dev)

        E_k = E0
        # [BUG-K3-FIX] bao gồm E^0 trong danh sách concat
        layer_outputs = [E0]

        for l in range(self.n_layers):
            nb = torch.sparse.mm(adj, E_k)

            if self.agg_type == "bi-interaction":
                E_k = (
                    F.leaky_relu(self.W_gc[l](E_k + nb))
                    + F.leaky_relu(self.W_bi[l](E_k * nb))
                )
            elif self.agg_type == "graphsage":
                E_k = F.leaky_relu(self.W_gc[l](torch.cat([E_k, nb], dim=-1)))
            else:  # gcn
                E_k = F.leaky_relu(self.W_gc[l](nb))

            E_k = F.normalize(E_k, dim=-1)
            layer_outputs.append(E_k)

        # [BUG-K3-FIX] concat (n_layers+1) outputs → W_out
        E_concat = torch.cat(layer_outputs, dim=-1)  # [N+M, (K+1)*D]
        E_final  = self.W_out(E_concat)              # [N+M, D]

        return E_final[: self.n_users], E_final[self.n_users :]

    # ── BaseModel interface ───────────────────────────────────────────────────

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        precomputed_entity_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entity_emb = (
            precomputed_entity_emb
            if precomputed_entity_emb is not None
            else self._compute_entity_embeddings()
        )
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
        self.kg_adj = kg_adj
        self._kg_tensors_cpu = None

    def to(self, *args, **kwargs):
        self._kg_tensors_cpu = None
        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        self._kg_tensors_cpu = None
        return super().cuda(device)

    def cpu(self):
        self._kg_tensors_cpu = None
        return super().cpu()
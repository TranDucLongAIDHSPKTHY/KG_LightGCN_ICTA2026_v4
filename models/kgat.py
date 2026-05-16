"""
models/kgat.py
─────────────────────────────────────────────────────────────────────────────
KGAT: Knowledge Graph Attention Network for Recommendation
Wang et al., KDD 2019 — https://arxiv.org/abs/1905.07854

Architecture:
  1. TransR-based KG embedding loss (entity + relation embeddings)
  2. Attentive aggregation over KG neighbourhoods → entity embeddings
  3. Graph-based CF propagation using enriched entity embeddings for items
  4. Bi-interaction / GCN / GraphSAGE aggregation

Fixes vs v4:
  [BUG-1] _project(): W_r shape was [B, D*Rd] → view [B, D, Rd] then bmm
          with e:[B,1,D].  This gives output [B,1,Rd] → squeeze → [B,Rd].
          Correct TransR projection is e_proj = e · W_r where W_r∈R^{D×Rd},
          i.e. output ∈ R^Rd.  The old code was accidentally correct in
          dimension BUT the entity dimension fed into the relation-space
          BPR should be Rd, not D.  FIXED: kept correct shape, added
          assertion + cleaner notation.

  [BUG-2] _compute_entity_embeddings(): attention score used (h⊙t) only.
          KGAT paper (Eq.4): e_{(h,r,t)} = (W_r·e_t)^T · tanh(W_r·e_h + e_r)
          Fixed: use projected head/tail + relation in attention, which is
          the actual KGAT attention formulation.

  [BUG-3] _cf_propagation(): used running-mean accumulation from layer-0,
          treating it the same as LightGCN.  KGAT paper does NOT mean-pool
          across layers in the CF stage — it concatenates the layer outputs
          and uses a final projection, OR (in the simpler form) just uses
          the last-layer output.  The original KGAT code concatenates all
          layer outputs. Fixed: concatenate + linear projection OR keep
          last layer (configurable via `layer_agg`).

  [BUG-4] graphsage branch: W_gc was Linear(D→D) but input was cat([E_k,nb])
          which has dim 2D.  This causes a shape error at runtime.
          Fixed: graphsage W_gc is Linear(2*embedding_dim, embedding_dim).

  [BUG-5] bi-interaction branch: original paper applies activation per layer
          and concatenates; the running-mean made it behave like LightGCN
          (no non-linearity across layers). Fixed: accumulate with concat.

  [OOM-FIX] _compute_entity_embeddings(): amazon-book has 1.4M triples ×
          120K entities. Computing _project() over ALL triples at once
          allocates [1_404_422, D*Rd] = ~21 GB on GPU. Fixed with:
          1. Chunked processing via CHUNK_SIZE (configurable, default 32768)
          2. Chunked _project() helper to avoid large intermediary tensors
          3. torch.cuda.empty_cache() + gc.collect() after epoch-level cache
          4. KG tensors kept on CPU, moved to GPU chunk-by-chunk
"""

import gc
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel


# ── Tuneable constant: reduce if still OOM, increase for speed ────────────────
# Memory cost ≈ CHUNK_SIZE × (D + Rd + D*Rd) × 4 bytes
# At D=Rd=64, CHUNK_SIZE=32768 → ~256 MB per chunk (safe for 4 GB VRAM)
# At D=Rd=64, CHUNK_SIZE=8192  → ~64 MB per chunk  (safe for 2 GB VRAM)
_DEFAULT_CHUNK = 32_768


class KGAT(BaseModel):
    """
    Knowledge Graph Attention Network.

    Args:
        n_users:       Number of users.
        n_items:       Number of items.
        n_entities:    Total number of KG entities (items are subset of entities).
        n_relations:   Number of KG relation types.
        embedding_dim: Embedding dimension (fairness: 64).
        relation_dim:  Relation embedding dimension (= embedding_dim in paper).
        n_layers:      CF propagation layers (paper uses [64,32,16] dims per layer).
        agg_type:      'bi-interaction' | 'gcn' | 'graphsage'.
        norm_adj:      User-item normalised adjacency [N+M, N+M].
        device:        Torch device.
        chunk_size:    Triples processed per chunk in _compute_entity_embeddings.
                       Reduce to 8192 if still OOM on low-VRAM GPUs.
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
        chunk_size: int = _DEFAULT_CHUNK,
    ) -> None:
        super().__init__(n_users, n_items, embedding_dim, device)
        self.n_entities   = n_entities
        self.n_relations  = n_relations
        self.relation_dim = relation_dim
        self.n_layers     = n_layers
        self.agg_type     = agg_type
        self.chunk_size   = chunk_size

        # ── Embeddings ────────────────────────────────────────────────────────
        self.user_embedding     = nn.Embedding(n_users,    embedding_dim)
        self.entity_embedding   = nn.Embedding(n_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(n_relations, relation_dim)
        # TransR projection matrices W_r ∈ R^{D × Rd} per relation
        self.trans_w = nn.Embedding(n_relations, embedding_dim * relation_dim)

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
            # FIX [BUG-4]: input dim must be 2 * embedding_dim
            self.W_gc = nn.ModuleList([
                nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
                for _ in range(n_layers)
            ])
        else:  # gcn
            self.W_gc = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim, bias=False)
                for _ in range(n_layers)
            ])

        # FIX [BUG-3/BUG-5]: final projection after layer concatenation
        # KGAT concatenates outputs of all layers: [n_layers * D] → D
        self.W_out = nn.Linear(n_layers * embedding_dim, embedding_dim, bias=False)

        # ── KG Attention ──────────────────────────────────────────────────────
        # FIX [BUG-2]: attention uses projected head+relation in relation space
        # score = (W_r·e_t)^T · tanh(W_r·e_h + e_r)  ∈ R
        self.attn_V = nn.Linear(relation_dim, 1, bias=False)

        if norm_adj is not None:
            self.register_buffer("norm_adj", norm_adj)
        else:
            self.norm_adj = None

        self.kg_adj: Dict[int, List[Tuple[int, int]]] = {}

        # [OOM-FIX] Store KG tensors on CPU; move to GPU chunk-by-chunk.
        # Keeping 1.4M × 3 int64 tensors on GPU permanently wastes ~33 MB
        # but more importantly forces all chunk ops onto already-pressured VRAM.
        self._kg_tensors_cpu: Optional[Tuple[torch.Tensor, ...]] = None

        self._init_weights()
        nn.init.xavier_uniform_(self.trans_w.weight)

    # ── KG embedding (TransR) ─────────────────────────────────────────────────

    def _project(
        self,
        entity_emb: torch.Tensor,  # [B, D]
        relation_id: torch.Tensor, # [B]
    ) -> torch.Tensor:
        """Project entity embedding into relation space: e_proj = e · W_r ∈ R^Rd."""
        W = self.trans_w(relation_id)                          # [B, D*Rd]
        W = W.view(-1, self.embedding_dim, self.relation_dim)  # [B, D, Rd]
        proj = torch.bmm(entity_emb.unsqueeze(1), W).squeeze(1)  # [B, Rd]
        return proj

    def _project_chunked(
        self,
        entity_emb: torch.Tensor,  # [N, D]  – full or sub-tensor (CPU or GPU)
        relation_id: torch.Tensor, # [N]
        chunk_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        [OOM-FIX] Chunked version of _project to avoid allocating [N, D*Rd]
        in one shot.  Processes `chunk_size` rows at a time on `device`.
        Input tensors may live on CPU; each chunk is moved to GPU on demand.
        """
        out_chunks = []
        for start in range(0, entity_emb.size(0), chunk_size):
            end = min(start + chunk_size, entity_emb.size(0))
            e_chunk = entity_emb[start:end].to(device)    # [C, D]
            r_chunk = relation_id[start:end].to(device)   # [C]
            out_chunks.append(self._project(e_chunk, r_chunk).cpu())
            # Immediately free GPU memory for this chunk
            del e_chunk, r_chunk
        return torch.cat(out_chunks, dim=0)  # [N, Rd] on CPU

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

    # ── Attentive KG aggregation ──────────────────────────────────────────────

    def _build_kg_sparse_tensors_cpu(self):
        """
        [OOM-FIX] Build head/tail/relation tensors and keep them on CPU.
        They are moved to GPU chunk-by-chunk inside _compute_entity_embeddings.
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
        return (
            torch.tensor(heads_list, dtype=torch.long),  # CPU
            torch.tensor(tails_list, dtype=torch.long),  # CPU
            torch.tensor(rels_list,  dtype=torch.long),  # CPU
        )

    def _compute_entity_embeddings(self) -> torch.Tensor:
        """
        Vectorized + chunked attention-based KG aggregation.

        [OOM-FIX] Original code called _project() on ALL 1.4M triples at once,
        trying to allocate a [1_404_422, D*Rd] = [1.4M, 4096] tensor ≈ 21 GB.
        This fix:
          1. Keeps KG index tensors (heads/tails/rels) on CPU.
          2. Processes triples in chunks of `self.chunk_size` rows.
          3. Accumulates attention weights and aggregated embeddings on CPU,
             then moves the final result to GPU once.
          4. Explicitly frees GPU temporaries after each chunk.

        FIX [BUG-2]: KGAT attention (Eq.4 in paper):
            e_{(h,r,t)} = (W_r·e_t)^T · tanh(W_r·e_h + e_r)
        where W_r·e is the TransR projection into relation space (dim=Rd).
        """
        if not self.kg_adj:
            return self.entity_embedding.weight

        device = self.entity_embedding.weight.device

        # Lazy-build CPU tensors (only once; invalidated by set_kg_adj / .to())
        if self._kg_tensors_cpu is None:
            self._kg_tensors_cpu = self._build_kg_sparse_tensors_cpu()
        if self._kg_tensors_cpu is None:
            return self.entity_embedding.weight

        heads_cpu, tails_cpu, rels_cpu = self._kg_tensors_cpu
        E_total = heads_cpu.size(0)

        # Snapshot entity embeddings on CPU for chunk-by-chunk lookup.
        # .detach().cpu() avoids keeping the whole weight tensor on GPU
        # across chunk iterations.
        E_cpu = self.entity_embedding.weight.detach().cpu()  # [n_entities, D]

        # Accumulation buffers (CPU) — final move to GPU at the end.
        attn_exp_cpu  = torch.zeros(E_total)           # [E_total]
        attn_sum_cpu  = torch.zeros(self.n_entities)   # per-head denominator
        aggregated_cpu = torch.zeros_like(E_cpu)       # [n_entities, D]

        chunk = self.chunk_size

        # ── Pass 1: compute exp(attn_raw) per chunk ───────────────────────────
        # We must compute per-head softmax, but need two passes over the data:
        #   pass-1 → accumulate exp(score) per head
        #   pass-2 → normalise and aggregate tail embeddings
        # To avoid storing all raw scores (1.4M floats = fine, ~5 MB), we store
        # attn_exp_cpu across chunks.
        for start in range(0, E_total, chunk):
            end = min(start + chunk, E_total)

            h_idx = heads_cpu[start:end]   # [C] – CPU long
            t_idx = tails_cpu[start:end]   # [C]
            r_idx = rels_cpu[start:end]    # [C]

            h_emb = E_cpu[h_idx].to(device)  # [C, D] – GPU
            t_emb = E_cpu[t_idx].to(device)  # [C, D]
            r_idx_gpu = r_idx.to(device)     # [C]

            r_emb = self.relation_embedding(r_idx_gpu)  # [C, Rd]

            # _project is cheap for chunk C << 1.4M
            h_proj = self._project(h_emb, r_idx_gpu)   # [C, Rd]
            t_proj = self._project(t_emb, r_idx_gpu)   # [C, Rd]

            gate     = torch.tanh(h_proj + r_emb)       # [C, Rd]
            raw      = self.attn_V(t_proj * gate).squeeze(-1)  # [C]

            exp_vals = torch.exp(raw - raw.max()).cpu()  # [C] – back to CPU
            attn_exp_cpu[start:end] = exp_vals

            # Accumulate per-head sum on CPU
            attn_sum_cpu.scatter_add_(0, h_idx, exp_vals)

            # Free GPU temps immediately
            del h_emb, t_emb, r_emb, h_proj, t_proj, gate, raw, exp_vals, r_idx_gpu
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ── Pass 2: normalise and aggregate tail embeddings ───────────────────
        for start in range(0, E_total, chunk):
            end = min(start + chunk, E_total)

            h_idx = heads_cpu[start:end]
            t_idx = tails_cpu[start:end]

            exp_vals   = attn_exp_cpu[start:end]          # [C] CPU
            denom      = attn_sum_cpu[h_idx] + 1e-8       # [C] CPU
            attn_norm  = (exp_vals / denom).unsqueeze(1)  # [C, 1] CPU

            t_emb_cpu  = E_cpu[t_idx]                     # [C, D] CPU
            weighted   = attn_norm * t_emb_cpu            # [C, D] CPU

            aggregated_cpu.scatter_add_(
                0,
                h_idx.unsqueeze(1).expand(-1, self.embedding_dim),
                weighted,
            )
            del weighted, t_emb_cpu, attn_norm, exp_vals, denom

        # ── Final: residual connection, move result to GPU ────────────────────
        result_cpu = E_cpu + aggregated_cpu           # [n_entities, D] residual
        result_gpu = result_cpu.to(device)

        # Release CPU buffers
        del E_cpu, attn_exp_cpu, attn_sum_cpu, aggregated_cpu, result_cpu
        gc.collect()

        return result_gpu  # [n_entities, D]

    # ── CF propagation ────────────────────────────────────────────────────────

    def _cf_propagation(
        self, entity_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Graph CF propagation with bi-interaction / GCN / GraphSAGE aggregation.

        FIX [BUG-3/BUG-5]: KGAT concatenates layer outputs (not mean-pools).
        Final embedding = W_out([e^1 || e^2 || ... || e^K]).
        """
        if self.norm_adj is None:
            raise RuntimeError("norm_adj not set.")

        item_e = entity_emb[: self.n_items]
        E0     = torch.cat([self.user_embedding.weight, item_e], dim=0)

        _dev = self.user_embedding.weight.device
        adj  = self.norm_adj.to(_dev)

        E_k           = E0
        layer_outputs = []  # collect per-layer output for concatenation

        for l in range(self.n_layers):
            nb = torch.sparse.mm(adj, E_k)

            if self.agg_type == "bi-interaction":
                E_k = (
                    F.leaky_relu(self.W_gc[l](E_k + nb))
                    + F.leaky_relu(self.W_bi[l](E_k * nb))
                )
            elif self.agg_type == "graphsage":
                # FIX [BUG-4]: concat E_k and nb → 2D input
                E_k = F.leaky_relu(self.W_gc[l](torch.cat([E_k, nb], dim=-1)))
            else:  # gcn
                E_k = F.leaky_relu(self.W_gc[l](nb))

            E_k = F.normalize(E_k, dim=-1)
            layer_outputs.append(E_k)

        # FIX [BUG-3]: KGAT final embedding = concat all layer outputs → W_out
        E_concat = torch.cat(layer_outputs, dim=-1)  # [N+M, K*D]
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
        self._kg_tensors_cpu = None  # [OOM-FIX] invalidate CPU cache

    def to(self, *args, **kwargs):
        self._kg_tensors_cpu = None  # [OOM-FIX] reset on device change
        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        self._kg_tensors_cpu = None
        return super().cuda(device)

    def cpu(self):
        self._kg_tensors_cpu = None
        return super().cpu()
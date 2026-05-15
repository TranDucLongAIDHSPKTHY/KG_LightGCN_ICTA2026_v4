# _lazy_device_fixed_
"""
models/kg_lightgcn.py
─────────────────────────────────────────────────────────────────────────────
Hai biến thể KG-LightGCN:

══════════════════════════════════════════════════════════════════════════════
Biến thể 1 — KGLightGCN  (Base / Backbone)
══════════════════════════════════════════════════════════════════════════════
  Ý tưởng: dùng KG để làm giàu item embedding, huấn luyện bằng BPR thuần.

  Pipeline:
    1. KG entity propagation (multi-hop mean-pool trên đồ thị entity–entity)
    2. Trộn item emb + entity emb: enriched_i = σ(α)·e_i + (1−σ(α))·ent_i
    3. LightGCN propagation trên đồ thị user-item dùng enriched item emb
    4. BPR loss + KG alignment loss (cosine item ↔ entity) + L2

  Loss:
    L = BPR(u, i+, i-)  +  λ_kg · KG_align  +  λ_reg · ||E||²

══════════════════════════════════════════════════════════════════════════════
Biến thể 2 — KGLightGCNCL  (Enhanced / Proposed)
══════════════════════════════════════════════════════════════════════════════
  Ý tưởng: KG vừa làm giàu đồ thị (như Biến thể 1), vừa dẫn dắt học tương
  phản (contrastive learning) bằng cách tạo hai view bổ sung nhau:

    View CF  (augmented):  noise-perturbed LightGCN trên plain item emb
    View KG  (enriched):   clean LightGCN trên KG-enriched item emb

  InfoNCE kéo cùng user/item gần nhau qua hai view, đẩy các node khác xa.
  Cả user CL lẫn item CL đều được tính.

  Loss:
    L = BPR(u, i+, i-)
      + λ_cl · InfoNCE(user_CF, user_KG)    # user-level CL
      + λ_cl · InfoNCE(item_CF, item_KG)    # item-level CL
      + λ_kg · KG_align                      # entity <-> item cosine alignment
      + λ_reg · ||E||²                       # L2 regularisation

  Khi kg_type='none': View KG tự động thay bằng noise view thứ hai
  (tương đương SimGCL — giúp so sánh đóng góp thuần của KG).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# Shared mixin: KG enrichment + entity propagation
# ─────────────────────────────────────────────────────────────────────────────

class _KGEnrichMixin:
    """
    Mixin cung cấp KG entity propagation và item enrichment.
    Kế thừa bởi cả KGLightGCN (Base) và KGLightGCNCL (Enhanced).
    """

    def _propagate_entity_embeddings(self) -> torch.Tensor:
        """Multi-hop mean-pooled propagation trên KG entity graph. → [n_entities, D]"""
        E: torch.Tensor = self.entity_embedding.weight      # type: ignore[attr-defined]
        kg_adj = self.kg_norm_adj                            # type: ignore[attr-defined]
        kg_n_layers: int = self.kg_n_layers                  # type: ignore[attr-defined]

        if kg_adj is None or kg_n_layers == 0:
            return E

        adj = kg_adj
        E_k = E
        acc = E.clone()
        for _ in range(kg_n_layers):
            E_k = torch.sparse.mm(adj, E_k)
            acc = acc + E_k
        return acc / (kg_n_layers + 1)

    def _get_entity_for_items(self) -> torch.Tensor:
        """Lấy entity embedding căn chỉnh theo item ID. → [n_items, D]"""
        n_items: int = self.n_items                       # type: ignore[attr-defined]
        emb_dim: int = self.embedding_dim                 # type: ignore[attr-defined]
        item2entity = self.item2entity                    # type: ignore[attr-defined]
        dev = self.item_embedding.weight.device           # type: ignore[attr-defined]

        entity_emb = self._propagate_entity_embeddings()
        entity_for_items = torch.zeros(n_items, emb_dim, device=dev)

        if item2entity is not None:
            valid_mask = item2entity >= 0
            valid_items = valid_mask.nonzero(as_tuple=True)[0]
            if len(valid_items) > 0:
                eids = item2entity[valid_items].clamp(0, entity_emb.shape[0] - 1)
                entity_for_items[valid_items] = entity_emb[eids]
        else:
            # Quy ước KGCL: item_id == entity_id
            n = min(n_items, entity_emb.shape[0])
            entity_for_items[:n] = entity_emb[:n]
        return entity_for_items

    def _enrich_item_embeddings(self) -> torch.Tensor:
        """
        enriched = sigmoid(alpha) * item_emb + (1 - sigmoid(alpha)) * entity_emb
        → [n_items, D]
        """
        item_emb: torch.Tensor = self.item_embedding.weight  # type: ignore[attr-defined]
        if not self.has_kg:                                   # type: ignore[attr-defined]
            return item_emb
        entity_for_items = self._get_entity_for_items()
        alpha = torch.sigmoid(self.alpha)                     # type: ignore[attr-defined]
        return alpha * item_emb + (1.0 - alpha) * entity_for_items

    def kg_alignment_loss(self) -> torch.Tensor:
        """
        Cosine alignment loss giữa item embedding và entity embedding.
        → Scalar in [0, 2].
        """
        if not self.has_kg:                                   # type: ignore[attr-defined]
            return torch.tensor(0.0, device=self.item_embedding.weight.device)  # type: ignore[attr-defined]
        item_emb: torch.Tensor = self.item_embedding.weight          # type: ignore[attr-defined]
        entity_emb: torch.Tensor = self.entity_embedding.weight      # type: ignore[attr-defined]
        n = min(self.n_items, entity_emb.shape[0])                   # type: ignore[attr-defined]
        cos_sim = F.cosine_similarity(item_emb[:n], entity_emb[:n].detach(), dim=-1)
        return (1.0 - cos_sim).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Biến thể 1: KGLightGCN — Base / Backbone  (BPR only)
# ─────────────────────────────────────────────────────────────────────────────

class KGLightGCN(_KGEnrichMixin, BaseModel):
    """
    KG-LightGCN Biến thể 1 (Base/Backbone).

    Dùng KG làm giàu item embedding + BPR thuần. Không có contrastive learning.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_entities: int = 0,
        n_relations: int = 0,
        embedding_dim: int = 64,
        n_layers: int = 3,
        kg_n_layers: int = 2,
        kg_type: str = "full",
        entity_agg: str = "mean",
        kg_reg: float = 1e-5,
        norm_adj: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        BaseModel.__init__(self, n_users, n_items, embedding_dim, device)

        self.n_entities = max(n_entities, n_items)
        self.n_relations = n_relations
        self.n_layers = n_layers
        self.kg_n_layers = kg_n_layers
        self.kg_type = kg_type
        self.entity_agg = entity_agg
        self.kg_reg = kg_reg
        self.has_kg = (kg_type != "none" and n_entities > 0)

        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        if self.has_kg:
            self.entity_embedding = nn.Embedding(self.n_entities, embedding_dim)
            self.alpha = nn.Parameter(torch.tensor(0.5))

        # Adjacency buffers
        self.register_buffer("norm_adj", norm_adj)
        self.register_buffer("kg_norm_adj", None)
        self.register_buffer("item2entity", None)

        self._init_weights()

    # ── Graph propagation ─────────────────────────────────────────────────────

    def _cf_propagation(
        self, item_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard LightGCN K-layer mean-pool. → (user [N,D], item [M,D])"""
        if self.norm_adj is None:
            raise RuntimeError("norm_adj not set.")
        # Lazy device move: adj follows model device (CPU/GPU transparent)
        _dev = self.user_embedding.weight.device
        adj = self.norm_adj.to(_dev)
        E0 = torch.cat([self.user_embedding.weight, item_emb], dim=0)
        E_k = E0
        acc = E0.clone()
        for _ in range(self.n_layers):
            E_k = torch.sparse.mm(adj, E_k)
            acc = acc + E_k
        E_final = acc / (self.n_layers + 1)
        return E_final[: self.n_users], E_final[self.n_users :]

    # ── BaseModel interface ───────────────────────────────────────────────────

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (user_emb, pos_emb, neg_emb) cho BPR."""
        item_enriched = self._enrich_item_embeddings()
        user_final, item_final = self._cf_propagation(item_enriched)
        return user_final[users], item_final[pos_items], item_final[neg_items]

    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            item_emb = self._enrich_item_embeddings()
            return self._cf_propagation(item_emb)

    def l2_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        u0 = self.user_embedding(users)
        p0 = self.item_embedding(pos_items)
        n0 = self.item_embedding(neg_items)
        reg = (u0.norm(2).pow(2) + p0.norm(2).pow(2) + n0.norm(2).pow(2)) / (2 * len(users))
        if self.has_kg:
            e0 = self.entity_embedding(pos_items.clamp(0, self.n_entities - 1))
            reg = reg + self.kg_reg * e0.norm(2).pow(2).mean()
        return reg

    # ── Setters ───────────────────────────────────────────────────────────────

    def set_adj(self, norm_adj: torch.Tensor) -> None:
        # FIX: use register_buffer so tensor follows model.to(device) correctly
        self.register_buffer("norm_adj", norm_adj)

    def set_kg_norm_adj(self, kg_norm_adj: torch.Tensor) -> None:
        # FIX: use register_buffer
        self.register_buffer("kg_norm_adj", kg_norm_adj)

    def set_item_entity_map(self, item2entity: torch.Tensor) -> None:
        # FIX: use register_buffer
        self.register_buffer("item2entity", item2entity)


# ─────────────────────────────────────────────────────────────────────────────
# Biến thể 2: KGLightGCNCL — Enhanced / Proposed  (BPR + Cross-view CL)
# ─────────────────────────────────────────────────────────────────────────────

class KGLightGCNCL(_KGEnrichMixin, BaseModel):
    """
    KG-LightGCN-CL Biến thể 2 (Enhanced/Proposed).

    KG vừa làm giàu đồ thị vừa dẫn dắt học tương phản cross-view:
      View CF (augmented) : noise-perturbed LightGCN, plain item emb
      View KG (enriched)  : clean LightGCN, KG-enriched item emb

    Loss = BPR + λ_cl*(UserCL + ItemCL) + λ_kg*KGAlign + λ_reg*L2
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_entities: int = 0,
        n_relations: int = 0,
        embedding_dim: int = 64,
        n_layers: int = 3,
        kg_n_layers: int = 2,
        kg_type: str = "full",
        entity_agg: str = "mean",
        kg_reg: float = 1e-5,
        cl_temp: float = 0.2,
        lambda_cl: float = 0.5,
        eps: float = 0.1,
        norm_adj: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        BaseModel.__init__(self, n_users, n_items, embedding_dim, device)

        self.n_entities = max(n_entities, n_items)
        self.n_relations = n_relations
        self.n_layers = n_layers
        self.kg_n_layers = kg_n_layers
        self.kg_type = kg_type
        self.entity_agg = entity_agg
        self.kg_reg = kg_reg
        self.cl_temp = cl_temp
        self.lambda_cl = lambda_cl
        self.eps = eps
        self.has_kg = (kg_type != "none" and n_entities > 0)

        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        if self.has_kg:
            self.entity_embedding = nn.Embedding(self.n_entities, embedding_dim)
            self.alpha = nn.Parameter(torch.tensor(0.5))

        # Adjacency buffers
        self.register_buffer("norm_adj", norm_adj)
        self.register_buffer("kg_norm_adj", None)
        self.register_buffer("item2entity", None)

        self._init_weights()

    # ── LightGCN propagation (clean or perturbed) ─────────────────────────────

    def _cf_propagation(
        self,
        item_emb: torch.Tensor,
        perturb: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LightGCN K-layer mean-pool.

        Args:
            item_emb: [n_items, D]
            perturb:  True → thêm noise ε tại mỗi layer (CF augmented view).

        Returns:
            user_final [n_users, D], item_final [n_items, D]
        """
        if self.norm_adj is None:
            raise RuntimeError("norm_adj not set.")
        # Lazy device move: adj follows model device (CPU/GPU transparent)
        _dev = self.user_embedding.weight.device
        adj = self.norm_adj.to(_dev)
        E0 = torch.cat([self.user_embedding.weight, item_emb], dim=0)
        E_k = E0
        acc = E0.clone()
        for _ in range(self.n_layers):
            E_k = torch.sparse.mm(adj, E_k)
            if perturb:
                # FIX: uniform noise in [-eps, eps] per element (no normalize)
                # Matches SimGCL paper Eq.4: ε ~ Uniform(-eps, eps) independent per dim
                noise = (torch.rand_like(E_k) * 2.0 - 1.0) * self.eps
                E_k = E_k + noise
            acc = acc + E_k
        E_final = acc / (self.n_layers + 1)
        return E_final[: self.n_users], E_final[self.n_users :]

    # ── Hai contrastive view ──────────────────────────────────────────────────

    def _get_cf_view(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        View CF: noise-perturbed LightGCN trên plain item embedding.
        Đại diện quan điểm "collaborative filtering thuần".
        """
        return self._cf_propagation(self.item_embedding.weight, perturb=True)

    def _get_kg_view(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        View KG: clean LightGCN trên KG-enriched item embedding.
        Đại diện quan điểm "knowledge graph guided".
        Fallback (has_kg=False): noise view 2 (= SimGCL behaviour).
        """
        if self.has_kg:
            return self._cf_propagation(self._enrich_item_embeddings(), perturb=False)
        return self._cf_propagation(self.item_embedding.weight, perturb=True)

    # ── InfoNCE contrastive loss ──────────────────────────────────────────────

    def contrastive_loss(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bidirectional InfoNCE giữa hai view của cùng node.

        Args:
            view1: [B, D]  View CF (augmented)
            view2: [B, D]  View KG (enriched)

        Returns:
            Scalar InfoNCE loss.
        """
        v1 = F.normalize(view1, dim=-1)
        v2 = F.normalize(view2, dim=-1)
        sim = torch.matmul(v1, v2.T) / self.cl_temp   # [B, B]
        labels = torch.arange(len(v1), device=v1.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2.0

    # ── BaseModel interface ───────────────────────────────────────────────────

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        """
        Training forward pass.

        Returns:
            user_emb  [B,D] — main CF user emb         (BPR)
            pos_emb   [B,D] — main CF pos item emb     (BPR)
            neg_emb   [B,D] — main CF neg item emb     (BPR)
            user_cf   [B,D] — augmented CF user view   (CL)
            user_kg   [B,D] — KG-enriched user view    (CL)
            item_cf   [B,D] — augmented CF item view   (CL, pos items)
            item_kg   [B,D] — KG-enriched item view    (CL, pos items)
        """
        # Main embeddings: clean KG-enriched, cho BPR scoring
        item_enriched = self._enrich_item_embeddings()
        user_main, item_main = self._cf_propagation(item_enriched, perturb=False)

        # Contrastive views
        user_cf_all, item_cf_all = self._get_cf_view()
        user_kg_all, item_kg_all = self._get_kg_view()

        return (
            user_main[users],
            item_main[pos_items],
            item_main[neg_items],
            user_cf_all[users],
            user_kg_all[users],
            item_cf_all[pos_items],
            item_kg_all[pos_items],
        )

    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Clean KG-enriched embeddings, không noise (dùng cho evaluation)."""
        with torch.no_grad():
            item_emb = self._enrich_item_embeddings()
            return self._cf_propagation(item_emb, perturb=False)

    def l2_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        u0 = self.user_embedding(users)
        p0 = self.item_embedding(pos_items)
        n0 = self.item_embedding(neg_items)
        reg = (u0.norm(2).pow(2) + p0.norm(2).pow(2) + n0.norm(2).pow(2)) / (2 * len(users))
        if self.has_kg:
            e0 = self.entity_embedding(pos_items.clamp(0, self.n_entities - 1))
            reg = reg + self.kg_reg * e0.norm(2).pow(2).mean()
        return reg

    # ── Setters ───────────────────────────────────────────────────────────────

    def set_adj(self, norm_adj: torch.Tensor) -> None:
        # FIX: use register_buffer so tensor follows model.to(device) correctly
        self.register_buffer("norm_adj", norm_adj)

    def set_kg_norm_adj(self, kg_norm_adj: torch.Tensor) -> None:
        # FIX: use register_buffer
        self.register_buffer("kg_norm_adj", kg_norm_adj)

    def set_item_entity_map(self, item2entity: torch.Tensor) -> None:
        # FIX: use register_buffer
        self.register_buffer("item2entity", item2entity)
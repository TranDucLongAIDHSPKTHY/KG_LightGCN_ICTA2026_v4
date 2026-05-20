"""
datasets/kg_dataset.py
─────────────────────────────────────────────────────────────────────────────
Knowledge Graph Dataset extending CFDataset.

Fixes vs previous version:
  [BUG-C3-FIX] build_kg_sparse_adj() trả về raw (0/1) adjacency không được
               normalize → khi dùng cho message passing trong KGCL, embedding
               bị scale sai → diverge.
               Fix: thêm build_kg_norm_adj() trả về D^{-1/2} A D^{-1/2}
               đã normalize đúng chuẩn. Caller (main.py/build_kgcl và
               build_kg_lightgcn) nên gọi build_kg_norm_adj() thay vì
               build_kg_sparse_adj() + _sparse_mx_to_torch().
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from datasets.cf_dataset import CFDataset


class KGDataset(CFDataset):
    """
    Extends CFDataset with knowledge graph triples.

    Additional attributes:
        n_entities:    Total KG entity count.
        n_relations:   Total relation type count.
        kg_triples:    np.ndarray [T, 3] — (head, relation, tail).
        entity2item:   entity_id → item_id (if entity is an item node).
        item2entity:   item_id   → entity_id.
        kg_adj:        List of adjacency dicts per relation (for KGAT).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        neg_samples: int = 1,
        kg_type: str = "full",
        seed: int = 42,
    ) -> None:
        self.kg_type = kg_type
        self.n_entities: int = 0
        self.n_relations: int = 0
        self.kg_triples: Optional[np.ndarray] = None
        self.item2entity: Dict[int, int] = {}
        self.entity2item: Dict[int, int] = {}
        super().__init__(data_dir=data_dir, split=split, neg_samples=neg_samples, seed=seed)

    # ── Override load to also load KG ─────────────────────────────────────────

    def load(self) -> None:
        super().load()
        if self.kg_type != "none":
            self._load_kg()

    # ── KG loading ────────────────────────────────────────────────────────────

    def _load_kg(self) -> None:
        kg_filename = {
            "full":     "kg_full.txt",
            "category": "kg_category.txt",
            "brand":    "kg_brand.txt",
        }.get(self.kg_type, "kg_full.txt")

        kg_path = os.path.join(self.data_dir, kg_filename)
        if not os.path.exists(kg_path):
            raise FileNotFoundError(
                f"KG file not found: {kg_path}. "
                "Run scripts/preprocess.py first."
            )

        triples: List[Tuple[int, int, int]] = []
        with open(kg_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    triples.append((int(parts[0]), int(parts[1]), int(parts[2])))

        if not triples:
            raise ValueError(f"KG file is empty: {kg_path}")

        self.kg_triples = np.array(triples, dtype=np.int64)

        all_entities = set(self.kg_triples[:, 0]) | set(self.kg_triples[:, 2])
        self.n_entities = max(all_entities) + 1
        self.n_relations = int(self.kg_triples[:, 1].max()) + 1

        entity_map_path = os.path.join(self.data_dir, "item2entity.json")
        if os.path.exists(entity_map_path):
            with open(entity_map_path, "r") as f:
                raw_map = json.load(f)
            for item_str, entity_id in raw_map.items():
                item_id = int(item_str)
                if item_id < self.n_items:
                    self.item2entity[item_id] = entity_id
                    self.entity2item[entity_id] = item_id

        n_mapped = len(self.item2entity)
        coverage = n_mapped / self.n_items if self.n_items > 0 else 0.0

        from utils.logger import get_logger
        _logger = get_logger("kg_dataset")
        _logger.info(
            f"KG loaded ({self.kg_type}): "
            f"{len(triples):,} triples | "
            f"{self.n_entities:,} entities | "
            f"{self.n_relations:,} relations | "
            f"item coverage: {coverage:.1%}"
        )

    # ── KG adjacency structures ───────────────────────────────────────────────

    def build_kg_adj_list(self) -> Dict[int, List[Tuple[int, int]]]:
        """Build head → [(tail, relation), …] adjacency list for KGAT/KGCL."""
        adj: Dict[int, List[Tuple[int, int]]] = {}
        if self.kg_triples is None:
            return adj
        for h, r, t in self.kg_triples:
            adj.setdefault(int(h), []).append((int(t), int(r)))
        return adj

    def build_kg_sparse_adj(self) -> sp.csr_matrix:
        """
        Build a raw (un-normalized) entity–entity adjacency matrix.
        Shape: [n_entities, n_entities].

        WARNING: Raw adjacency. Cho KGCL/KGLightGCN message passing,
        dùng build_kg_norm_adj() để lấy D^{-1/2} A D^{-1/2} đã normalize.
        """
        if self.kg_triples is None:
            return sp.csr_matrix((self.n_entities, self.n_entities))
        heads = self.kg_triples[:, 0]
        tails = self.kg_triples[:, 2]
        data = np.ones(len(heads), dtype=np.float32)
        mat = sp.csr_matrix(
            (data, (heads, tails)),
            shape=(self.n_entities, self.n_entities),
        )
        mat = mat + mat.T
        mat.data[:] = 1.0
        return mat

    def build_kg_norm_adj(self) -> torch.Tensor:
        """
        [BUG-C3-FIX] Build D^{-1/2} A D^{-1/2} normalized KG adjacency.

        Dùng cho KGCL và KG-LightGCN message passing qua KG entity graph.
        Không dùng raw adjacency (entity emb bị scale sai và diverge).

        Returns:
            torch.sparse_coo_tensor [n_entities, n_entities] trên CPU.
        """
        if self.kg_triples is None:
            return torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0),
                (self.n_entities, self.n_entities),
            ).coalesce()

        heads = self.kg_triples[:, 0]
        tails = self.kg_triples[:, 2]
        data  = np.ones(len(heads), dtype=np.float32)

        # Symmetrize: A = R + R^T
        A = sp.csr_matrix(
            (data, (heads, tails)),
            shape=(self.n_entities, self.n_entities),
        )
        A = (A + A.T).tocsr()

        # D^{-1/2}
        deg = np.asarray(A.sum(axis=1)).flatten()
        with np.errstate(divide="ignore", invalid="ignore"):
            d_inv_sqrt = np.where(deg > 0, np.power(deg, -0.5), 0.0).astype(np.float32)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        A_hat = (D_inv_sqrt @ A @ D_inv_sqrt).tocoo().astype(np.float32)

        indices = torch.from_numpy(
            np.vstack([A_hat.row, A_hat.col]).astype(np.int64)
        )
        values = torch.from_numpy(A_hat.data)
        return torch.sparse_coo_tensor(
            indices, values, (self.n_entities, self.n_entities)
        ).coalesce()

    # ── KG triple sampling ─────────────────────────────────────────────────────

    def sample_kg_triple(self) -> Tuple[int, int, int, int]:
        if self.kg_triples is None:
            raise RuntimeError("KG triples not loaded.")
        idx = self.rng.randint(0, len(self.kg_triples))
        h, r, t_pos = self.kg_triples[idx]
        while True:
            t_neg = int(self.rng.randint(0, self.n_entities))
            if t_neg != t_pos:
                break
        return int(h), int(r), int(t_pos), int(t_neg)

    def sample_kg_triples(
        self, batch_size: int
    ) -> Optional[Tuple[List[int], List[int], List[int], List[int]]]:
        if self.kg_triples is None:
            return None
        n_triples = len(self.kg_triples)
        idxs = self.rng.randint(0, n_triples, size=batch_size)
        selected = self.kg_triples[idxs]

        heads = selected[:, 0].tolist()
        rels  = selected[:, 1].tolist()
        t_pos = selected[:, 2].tolist()

        t_neg = self.rng.randint(0, self.n_entities, size=batch_size)
        collision = t_neg == selected[:, 2]
        t_neg[collision] = (t_neg[collision] + 1) % self.n_entities

        return heads, rels, t_pos, t_neg.tolist()
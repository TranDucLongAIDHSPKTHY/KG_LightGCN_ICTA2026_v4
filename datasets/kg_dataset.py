"""
datasets/kg_dataset.py
─────────────────────────────────────────────────────────────────────────────
Knowledge Graph Dataset extending CFDataset.
Adds KG triple loading, entity/relation mappings, and KG adjacency structures
needed by KGAT, KGCL, and KG-LightGCN.
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
        """
        Args:
            data_dir:    Path to processed dataset directory.
            split:       'train' | 'val' | 'test'.
            neg_samples: Negatives per positive for BPR.
            kg_type:     'full' | 'category' | 'brand' | 'none'.
            seed:        RNG seed.
        """
        self.kg_type = kg_type
        self.n_entities: int = 0
        self.n_relations: int = 0
        self.kg_triples: Optional[np.ndarray] = None
        self.item2entity: Dict[int, int] = {}
        self.entity2item: Dict[int, int] = {}
        super().__init__(data_dir=data_dir, split=split, neg_samples=neg_samples, seed=seed)

    # ── Override load to also load KG ─────────────────────────────────────────

    def load(self) -> None:
        """Load CF data (via parent) then load KG."""
        super().load()
        if self.kg_type != "none":
            self._load_kg()

    # ── KG loading ────────────────────────────────────────────────────────────

    def _load_kg(self) -> None:
        """Load KG triples and entity map from disk."""
        # Support KGAT repo (kg_full.txt built by preprocess.py from kg_final.txt)
        kg_filename = {
            "full":     "kg_full.txt",
            "category": "kg_category.txt",
            "brand":    "kg_brand.txt",
        }.get(self.kg_type, "kg_full.txt")

        kg_path = os.path.join(self.data_dir, kg_filename)
        if not os.path.exists(kg_path):
            raise FileNotFoundError(
                f"KG file not found: {kg_path}. "
                "Run scripts/preprocess.py first (Amazon-Book required for KG)."
            )

        # Read triples
        triples: List[Tuple[int, int, int]] = []
        with open(kg_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    triples.append((int(parts[0]), int(parts[1]), int(parts[2])))

        if not triples:
            raise ValueError(f"KG file is empty: {kg_path}")

        self.kg_triples = np.array(triples, dtype=np.int64)

        # Entity/relation counts
        all_entities = set(self.kg_triples[:, 0]) | set(self.kg_triples[:, 2])
        self.n_entities = max(all_entities) + 1
        self.n_relations = int(self.kg_triples[:, 1].max()) + 1

        # Load entity→item mapping (saved by preprocess.py)
        entity_map_path = os.path.join(self.data_dir, "item2entity.json")
        if os.path.exists(entity_map_path):
            with open(entity_map_path, "r") as f:
                raw_map = json.load(f)
            # item2entity.json format (KGAT repo):
            #   key = new_item_id (str), value = kg_entity_id
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
        """
        Build head → [(tail, relation), …] adjacency list for KGAT/KGCL.
        """
        adj: Dict[int, List[Tuple[int, int]]] = {}
        if self.kg_triples is None:
            return adj
        for h, r, t in self.kg_triples:
            adj.setdefault(int(h), []).append((int(t), int(r)))
        return adj

    def build_kg_sparse_adj(self) -> sp.csr_matrix:
        """
        Build a sparse entity–entity adjacency matrix from KG triples.
        Shape: [n_entities, n_entities].
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
        # Symmetrize
        mat = mat + mat.T
        mat.data[:] = 1.0
        return mat

    # ── KG triple sampling (for TransR / KG embedding losses) ─────────────────

    def sample_kg_triple(self) -> Tuple[int, int, int, int]:
        """
        Sample one positive KG triple and one corrupted (negative) triple.

        Returns:
            (head, relation, pos_tail, neg_tail)
        """
        if self.kg_triples is None:
            raise RuntimeError("KG triples not loaded.")
        idx = self.rng.integers(0, len(self.kg_triples))
        h, r, t_pos = self.kg_triples[idx]
        # Corrupt tail
        while True:
            t_neg = int(self.rng.integers(0, self.n_entities))
            if t_neg != t_pos:
                break
        return int(h), int(r), int(t_pos), int(t_neg)

    def sample_kg_triples(
        self, batch_size: int
    ) -> Optional[Tuple[List[int], List[int], List[int], List[int]]]:
        """
        Sample a batch of KG triples (positive + corrupted negative) efficiently.

        Args:
            batch_size: Number of triples to sample.

        Returns:
            (heads, relations, pos_tails, neg_tails) each as a list of ints,
            or None if KG triples are not loaded.
        """
        if self.kg_triples is None:
            return None
        n_triples = len(self.kg_triples)
        idxs = self.rng.integers(0, n_triples, size=batch_size)
        selected = self.kg_triples[idxs]  # [B, 3]

        heads    = selected[:, 0].tolist()
        rels     = selected[:, 1].tolist()
        t_pos    = selected[:, 2].tolist()

        # Corrupt tails (vectorized)
        t_neg = self.rng.integers(0, self.n_entities, size=batch_size)
        # Ensure neg != pos (fix collisions)
        collision = t_neg == selected[:, 2]
        t_neg[collision] = (t_neg[collision] + 1) % self.n_entities

        return heads, rels, t_pos, t_neg.tolist()

    # ── Dataset interface (re-use parent __getitem__) ─────────────────────────
    # KGDataset inherits BPR sampling from CFDataset.__getitem__
    # KG-specific trainers call sample_kg_triple() directly.

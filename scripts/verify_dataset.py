"""
scripts/verify_dataset.py
─────────────────────────────────────────────────────────────────────────────
Script kiểm tra dataset statistics để đảm bảo preprocessing đúng.

Kiểm tra:
  1. n_users, n_items, n_interactions theo từng split
  2. Overlap giữa train/val/test
  3. Distribution interactions per user
  4. KG statistics (nếu có)
  5. So sánh với paper-reported numbers

Paper-reported Amazon-Book (LightGCN):
  #Users: 52,643
  #Items: 91,599
  #Interactions: 2,984,108
  Density: 0.00062

Paper-reported Amazon-Book (KGCL):
  #Users: 70,679
  #Items: 24,915
  #Interactions: 847,733

NOTE: Hai bộ số khác nhau vì LightGCN và KGCL dùng khác nhau:
  - LightGCN repo: raw Amazon-Book data
  - KGCL repo: ripple/kgcn-style split (smaller subset)
  Dự án này dùng LightGCN repo → target LightGCN paper numbers.

Usage:
  python scripts/verify_dataset.py --data_dir /data/phuongtran/processed/amazon-book
"""

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Paper targets (LightGCN repo, Amazon-Book)
LIGHTGCN_TARGETS = {
    "n_users": 52643,
    "n_items": 91599,
    "n_train": None,   # not explicitly stated
    "n_test":  None,
    "density": 0.00062,
}

# SimGCL paper Amazon-Book (dùng LightGCN repo, similar)
SIMGCL_TARGETS = {
    "recall@20": 0.0515,
    "ndcg@20": 0.0414,
}

KGCL_TARGETS = {
    "recall@20": 0.0889,
    "ndcg@20": 0.0584,
}


def read_interaction_file(path: str) -> Dict[int, List[int]]:
    user2items: Dict[int, List[int]] = {}
    if not os.path.exists(path):
        return user2items
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            uid = int(parts[0])
            user2items[uid] = [int(x) for x in parts[1:]]
    return user2items


def analyze_split(
    user2items: Dict[int, List[int]],
    name: str,
) -> Dict:
    if not user2items:
        print(f"  {name}: EMPTY or not found")
        return {}

    n_users = len(user2items)
    n_interactions = sum(len(v) for v in user2items.values())
    all_items = {i for v in user2items.values() for i in v}
    n_items = len(all_items)

    user_sizes = sorted(len(v) for v in user2items.values())
    avg_per_user = n_interactions / n_users if n_users else 0

    # Distribution
    size_counter = Counter(len(v) for v in user2items.values())
    n_single = size_counter.get(1, 0)

    print(f"  {name}:")
    print(f"    n_users        = {n_users:,}")
    print(f"    n_items        = {n_items:,}")
    print(f"    n_interactions = {n_interactions:,}")
    print(f"    avg per user   = {avg_per_user:.2f}")
    print(f"    min/max/median = {user_sizes[0]}/{user_sizes[-1]}/{user_sizes[len(user_sizes)//2]}")
    print(f"    users with 1 item = {n_single:,} ({n_single/n_users*100:.1f}%)")

    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "avg_per_user": avg_per_user,
        "n_single_item_users": n_single,
    }


def check_5core(train_d: Dict, test_d: Dict) -> None:
    """Verify 5-core property."""
    print("\n[5-core check]")
    from collections import Counter as C
    all_pairs = []
    for u, items in train_d.items():
        for i in items:
            all_pairs.append((u, i))
    for u, items in test_d.items():
        for i in items:
            all_pairs.append((u, i))

    user_cnt = C(u for u, i in all_pairs)
    item_cnt = C(i for u, i in all_pairs)

    min_user = min(user_cnt.values()) if user_cnt else 0
    min_item = min(item_cnt.values()) if item_cnt else 0
    print(f"  Min user interactions (train+test): {min_user}")
    print(f"  Min item interactions (train+test): {min_item}")
    if min_user >= 5 and min_item >= 5:
        print("  ✓ 5-core constraint satisfied")
    else:
        print("  ✗ 5-core constraint NOT satisfied — may need reprocessing")


def check_overlap(
    train_d: Dict, val_d: Dict, test_d: Dict
) -> None:
    """Check item overlap between splits."""
    print("\n[Overlap check]")
    train_items = {i for v in train_d.values() for i in v}
    val_items   = {i for v in val_d.values()   for i in v}
    test_items  = {i for v in test_d.values()  for i in v}

    # Items in test but NOT in train (truly cold)
    cold_items = test_items - train_items
    print(f"  Test items not in train (truly cold): {len(cold_items):,}")
    print(f"  Val items:   {len(val_items):,}")
    print(f"  Test items:  {len(test_items):,}")
    print(f"  Train items: {len(train_items):,}")

    # User overlap
    train_users = set(train_d)
    test_users  = set(test_d)
    val_users   = set(val_d)
    print(f"  Test users not in train: {len(test_users - train_users):,}")
    print(f"  Val users not in train: {len(val_users - train_users):,}")


def compare_with_targets(
    train_d: Dict, val_d: Dict, test_d: Dict, stats_path: str
) -> None:
    """Compare with paper-reported numbers."""
    print("\n[Paper comparison — LightGCN Amazon-Book]")

    all_users = set(train_d) | set(val_d) | set(test_d)
    all_items = (
        {i for v in train_d.values() for i in v}
        | {i for v in val_d.values()   for i in v}
        | {i for v in test_d.values()  for i in v}
    )
    n_users = len(all_users)
    n_items = len(all_items)
    n_train = sum(len(v) for v in train_d.values())
    n_total = (
        n_train
        + sum(len(v) for v in val_d.values())
        + sum(len(v) for v in test_d.values())
    )
    density = n_total / (n_users * n_items) if n_users * n_items else 0

    target_u = LIGHTGCN_TARGETS["n_users"]
    target_i = LIGHTGCN_TARGETS["n_items"]

    print(f"  n_users  : {n_users:,} (target: {target_u:,}, diff: {n_users-target_u:+,})")
    print(f"  n_items  : {n_items:,} (target: {target_i:,}, diff: {n_items-target_i:+,})")
    print(f"  n_total  : {n_total:,}")
    print(f"  n_train  : {n_train:,}")
    print(f"  density  : {density:.5f} (target: ~0.00062)")

    if abs(n_users - target_u) > 1000 or abs(n_items - target_i) > 1000:
        print("  ⚠ WARNING: Dataset size differs significantly from LightGCN paper")
        print("    → Possible cause: different filtering (5-core vs 10-core)")
        print("    → Or: preprocessing applied to already-filtered LightGCN repo data")
    else:
        print("  ✓ Dataset size matches LightGCN paper")

    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"\n  Stats from stats.json:")
        for k in ["n_users", "n_items", "n_train", "n_val", "n_test",
                   "density_train", "density_all_splits", "kg_coverage"]:
            if k in stats:
                print(f"    {k}: {stats[k]}")


def check_kg(data_dir: str) -> None:
    """Check KG file statistics."""
    kg_path = os.path.join(data_dir, "kg_full.txt")
    if not os.path.exists(kg_path):
        print("\n[KG check] kg_full.txt not found")
        return

    print("\n[KG check]")
    triples = []
    with open(kg_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                triples.append((int(parts[0]), int(parts[1]), int(parts[2])))

    if not triples:
        print("  kg_full.txt is empty!")
        return

    heads = [h for h, r, t in triples]
    rels  = [r for h, r, t in triples]
    tails = [t for h, r, t in triples]

    print(f"  n_triples  : {len(triples):,}")
    print(f"  n_relations: {max(rels)+1}")
    print(f"  max_entity : {max(max(heads), max(tails))}")
    print(f"  rel distribution:")
    rel_count = Counter(rels)
    for r in sorted(rel_count):
        print(f"    rel {r}: {rel_count[r]:,} triples")

    # Check item2entity
    entity_path = os.path.join(data_dir, "item2entity.json")
    if os.path.exists(entity_path):
        with open(entity_path) as f:
            i2e = json.load(f)
        print(f"  item2entity: {len(i2e):,} mappings")
    else:
        print("  item2entity.json: NOT FOUND")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default="/data/phuongtran/processed/amazon-book")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir

    print("=" * 65)
    print(f"DATASET VERIFICATION: {data_dir}")
    print("=" * 65)

    train_d = read_interaction_file(os.path.join(data_dir, "train.txt"))
    val_d   = read_interaction_file(os.path.join(data_dir, "val.txt"))
    test_d  = read_interaction_file(os.path.join(data_dir, "test.txt"))

    print("\n[Split statistics]")
    train_stats = analyze_split(train_d, "train")
    val_stats   = analyze_split(val_d,   "val")
    test_stats  = analyze_split(test_d,  "test")

    check_5core(train_d, test_d)
    check_overlap(train_d, val_d, test_d)
    compare_with_targets(train_d, val_d, test_d,
                         os.path.join(data_dir, "stats.json"))
    check_kg(data_dir)

    print("\n" + "=" * 65)
    print("VERIFICATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
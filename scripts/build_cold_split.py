"""
scripts/build_cold_split.py
─────────────────────────────────────────────────────────────────────────────
Cold-start split builder.

Protocol đúng:
  cold_items = X% items được sample từ TẤT CẢ items, ưu tiên:
    - items chỉ xuất hiện trong test (cold items thực sự)
    - nếu không đủ: items xuất hiện trong test với ít lượt train nhất

  Train_cold = {(u,i) ∈ train | i ∉ cold_items}
  Val_cold   = {(u,i) ∈ val   | i ∉ cold_items}
  Test       = giữ nguyên (cold items xuất hiện ở đây để evaluate)

  cold_test  = {(u,i) ∈ test | i ∈ cold_items}  ← dùng khi evaluate cold

Lưu ý:
  - Trong LightGCN dataset, hầu hết items xuất hiện cả trong train lẫn test
    (không có items "purely cold"). Do đó, ta relaxe constraint:
    cold_items = items ít xuất hiện nhất trong train (bottom X%)
    → Mô phỏng kịch bản long-tail / ít tương tác
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_script_logger
from utils.seed import set_seed

logger = get_script_logger("build_cold_split")

COLD_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_interaction_file(path: str) -> Dict[int, List[int]]:
    user2items: Dict[int, List[int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                uid = int(parts[0])
                user2items[uid] = [int(x) for x in parts[1:]]
            except ValueError:
                continue
    return user2items


def write_interaction_file(path: str, user2items: Dict[int, List[int]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for uid in sorted(user2items.keys()):
            items = user2items[uid]
            if items:
                f.write(f"{uid} " + " ".join(map(str, items)) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Cold split builder
# ─────────────────────────────────────────────────────────────────────────────

def build_cold_split(
    processed_dir: str,
    ratio: int,
    seed: int = COLD_SEED,
) -> None:
    """
    Build cold split bằng cách chọn X% items ít xuất hiện nhất trong train.
    Các items này sẽ bị xóa khỏi train/val → model không thấy chúng khi train.
    Chúng vẫn xuất hiện trong test → evaluate cold-start performance.

    Strategy:
      1. Tính tần suất của mỗi item trong train
      2. Chọn X% items có ít tương tác train nhất (long-tail items)
      3. Ưu tiên items CŨNG xuất hiện trong test (có thể evaluate được)
    """
    logger.info(f"  Building Cold-{ratio} split  (seed={seed}) ...")

    train_path = os.path.join(processed_dir, "train.txt")
    val_path   = os.path.join(processed_dir, "val.txt")
    test_path  = os.path.join(processed_dir, "test.txt")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Processed file not found: {p}. Run preprocess.py first."
            )

    user2train = read_interaction_file(train_path)
    user2val   = read_interaction_file(val_path)
    user2test  = read_interaction_file(test_path)

    # Tất cả items
    train_items: Set[int] = {i for v in user2train.values() for i in v}
    val_items:   Set[int] = {i for v in user2val.values()   for i in v}
    test_items:  Set[int] = {i for v in user2test.values()  for i in v}
    all_items = train_items | val_items | test_items

    # Tần suất item trong train
    train_freq = Counter(i for items in user2train.values() for i in items)

    n_cold = max(1, int(len(all_items) * ratio / 100))

    # ── Chiến lược chọn cold items ────────────────────────────────────────────
    # Ưu tiên 1: items trong test nhưng KHÔNG trong train (truly cold)
    truly_cold = list(test_items - train_items - val_items)
    # Ưu tiên 2: items trong test có ít train interactions nhất
    in_test_and_train = sorted(
        test_items & train_items,
        key=lambda i: train_freq.get(i, 0)   # ascending: ít nhất trước
    )
    # Ưu tiên 3: items chỉ trong train, sorted by freq
    train_only = sorted(
        train_items - test_items,
        key=lambda i: train_freq.get(i, 0)
    )

    # Ghép theo thứ tự ưu tiên
    candidate_pool = truly_cold + in_test_and_train + train_only

    # Shuffle trong mỗi nhóm để tránh deterministic bias, nhưng giữ thứ tự ưu tiên
    rng = random.Random(seed)
    rng.shuffle(truly_cold)
    rng.shuffle(in_test_and_train[:len(in_test_and_train)//2])   # shuffle nửa đầu (ít freq nhất)

    candidate_pool = truly_cold + in_test_and_train + train_only
    # Bỏ duplicate
    seen = set()
    ordered_candidates = []
    for i in candidate_pool:
        if i not in seen:
            seen.add(i)
            ordered_candidates.append(i)

    cold_items: Set[int] = set(ordered_candidates[:n_cold])

    # Thống kê
    cold_in_test  = cold_items & test_items
    cold_in_train = cold_items & train_items
    cold_in_val   = cold_items & val_items

    logger.info(
        f"  Cold items: {len(cold_items):,} / {len(all_items):,} "
        f"({len(cold_items)/len(all_items):.1%})"
    )
    logger.info(f"    In test:  {len(cold_in_test):,}  (evaluatable)")
    logger.info(f"    In train: {len(cold_in_train):,}  (will be removed)")
    logger.info(f"    In val:   {len(cold_in_val):,}    (will be removed)")

    if len(cold_in_test) == 0:
        logger.warning(
            "  ⚠  0 cold items appear in test! "
            "Cold evaluation will be empty. "
            "Check dataset — all test items may already be in train."
        )

    # Build filtered splits
    def filter_split(d: Dict[int, List[int]]) -> Dict[int, List[int]]:
        result = {}
        for uid, items in d.items():
            filtered = [i for i in items if i not in cold_items]
            if filtered:
                result[uid] = filtered
        return result

    train_cold = filter_split(user2train)
    val_cold   = filter_split(user2val)
    test_cold  = user2test   # Test KHÔNG thay đổi

    # Save
    out_dir = os.path.join(processed_dir, f"cold_{ratio}")
    os.makedirs(out_dir, exist_ok=True)

    write_interaction_file(os.path.join(out_dir, "train.txt"), train_cold)
    write_interaction_file(os.path.join(out_dir, "val.txt"),   val_cold)
    write_interaction_file(os.path.join(out_dir, "test.txt"),  test_cold)

    with open(os.path.join(out_dir, "cold_items.txt"), "w") as f:
        for iid in sorted(cold_items):
            f.write(f"{iid}\n")

    n_train = sum(len(v) for v in train_cold.values())
    n_val   = sum(len(v) for v in val_cold.values())
    n_test  = sum(len(v) for v in test_cold.values())
    cold_test_pairs = sum(
        1 for items in test_cold.values() for i in items if i in cold_items
    )

    stats = {
        "ratio": ratio,
        "n_cold_items": len(cold_items),
        "n_cold_in_test": len(cold_in_test),
        "n_cold_in_train_removed": len(cold_in_train),
        "n_all_items": len(all_items),
        "cold_fraction": round(len(cold_items) / len(all_items), 4),
        "train_interactions_after": n_train,
        "val_interactions_after": n_val,
        "test_interactions": n_test,
        "cold_test_pairs": cold_test_pairs,
        "seed": seed,
        "strategy": "least_frequent_train_items_preferring_test_items",
    }
    with open(os.path.join(out_dir, "cold_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"  Cold test pairs for evaluation: {cold_test_pairs:,}")
    logger.info(f"  ✓ Cold-{ratio} saved to: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cold-start splits")
    parser.add_argument(
        "--dataset", choices=["amazon-book", "yelp2018", "all"], default="all"
    )
    parser.add_argument(
        "--ratio", nargs="+", type=int, default=[10, 20, 30]
    )
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=COLD_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    datasets = ["amazon-book", "yelp2018"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        processed_dir = os.path.join(args.data_dir, ds)
        if not os.path.isdir(processed_dir):
            logger.error(f"Processed directory not found: {processed_dir}")
            continue
        logger.info(f"\nDataset: {ds}")
        for ratio in args.ratio:
            build_cold_split(processed_dir, ratio=ratio, seed=args.seed)

    logger.info("\nCold split generation complete.")


if __name__ == "__main__":
    main()

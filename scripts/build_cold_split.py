"""
scripts/build_cold_split.py
------------------------------------------------------------------------------
Cold-start split builder — TRUE COLD-START protocol.

ĐỊNH NGHĨA COLD-START (True Cold Items):
────────────────────────────────────────
  Cold items = items KHÔNG xuất hiện trong training set,
               chỉ xuất hiện trong test set.

  Protocol:
    1. Lấy tất cả items có tương tác trong test.
    2. Ngẫu nhiên chọn X% trong số đó làm cold_items (seed=42).
    3. Xóa toàn bộ tương tác của cold_items khỏi train và val.
    4. Giữ nguyên test set (để có ground truth cho cold items).
    5. KG (nếu có): giữ triples liên quan đến cold items.

3 MỨC:
  Cold-10 : 10% test items được chọn làm cold
  Cold-20 : 20% test items được chọn làm cold  
  Cold-30 : 30% test items được chọn làm cold

METRICS:
  HR@10_cold, NDCG@10_cold, Recall@20_cold

SEED: 42 (để tái lập)

Đầu ra mỗi ratio:
  data/processed/<dataset>/cold_<ratio>/train.txt
  data/processed/<dataset>/cold_<ratio>/val.txt
  data/processed/<dataset>/cold_<ratio>/test.txt
  data/processed/<dataset>/cold_<ratio>/kg_full.txt    (Amazon-Book)
  data/processed/<dataset>/cold_<ratio>/cold_items.txt
  data/processed/<dataset>/cold_<ratio>/cold_stats.json
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_script_logger
from utils.seed import set_seed

logger = get_script_logger("build_cold_split")

COLD_SEED   = 42
DATASETS_KG = {"amazon-book"}


# ------------------------------------------------------------------------------
# I/O helpers (giữ nguyên)
# ------------------------------------------------------------------------------

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


def read_kg_file(path: str) -> List[Tuple[int, int, int]]:
    triples: List[Tuple[int, int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            try:
                triples.append((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError:
                continue
    return triples


def write_kg_file(path: str, triples: List[Tuple[int, int, int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in sorted(triples):
            f.write(f"{h}\t{r}\t{t}\n")


# ------------------------------------------------------------------------------
# True Cold Item Selection
# ------------------------------------------------------------------------------

def select_true_cold_items(
    test_d:    Dict[int, List[int]],
    all_items: Set[int],
    ratio:     int,
    seed:      int = COLD_SEED,
) -> Tuple[Set[int], Dict]:
    """
    Chọn true cold items: items chỉ xuất hiện trong test.
    """
    # Lấy tất cả items có trong test
    test_items = {i for items in test_d.values() for i in items}
    
    if not test_items:
        raise ValueError("Test set rỗng!")

    n_cold = max(1, int(len(test_items) * ratio / 100))

    # Random chọn với seed cố định
    rng = random.Random(seed)
    test_item_list = sorted(test_items)          # sort để deterministic trước khi shuffle
    rng.shuffle(test_item_list)
    
    cold_items = set(test_item_list[:n_cold])

    info = {
        "n_cold":          len(cold_items),
        "n_test_items":    len(test_items),
        "n_all_items":     len(all_items),
        "target_fraction": round(ratio / 100, 4),
        "actual_fraction": round(len(cold_items) / len(test_items), 4),
        "seed":            seed,
    }

    logger.info(
        f"    Cold items   : {len(cold_items):,} / {len(test_items):,} test items "
        f"({len(cold_items)/len(test_items):.1%})"
    )
    logger.info(
        f"    Actual % of all items: {len(cold_items)/len(all_items):.2%}"
    )

    return cold_items, info


# ------------------------------------------------------------------------------
# KG filtering (giữ nguyên)
# ------------------------------------------------------------------------------

def filter_kg_by_cold_items(
    all_triples: List[Tuple[int, int, int]],
    cold_items:  Set[int],
) -> List[Tuple[int, int, int]]:
    if not cold_items:
        return []
    return [
        (h, r, t) for h, r, t in all_triples
        if h in cold_items or t in cold_items
    ]


# ------------------------------------------------------------------------------
# Core split builder (đã chỉnh sửa)
# ------------------------------------------------------------------------------

def build_cold_split(
    processed_dir: str,
    dataset:       str,
    ratio:         int,
    seed:          int = COLD_SEED,
) -> None:
    logger.info(
        f"\n{'─'*70}\n"
        f"  Building TRUE Cold-{ratio} | {dataset} | seed={seed}\n"
        f"  Protocol: True cold-start (items only in test, removed from train/val)\n"
        f"{'─'*70}"
    )

    # Đường dẫn
    train_path = os.path.join(processed_dir, "train.txt")
    val_path   = os.path.join(processed_dir, "val.txt")
    test_path  = os.path.join(processed_dir, "test.txt")
    kg_path    = os.path.join(processed_dir, "kg_full.txt")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Không tìm thấy: {p}. Chạy preprocess.py trước.")

    has_kg = (dataset in DATASETS_KG) and os.path.exists(kg_path)

    # Đọc data
    user2train = read_interaction_file(train_path)
    user2val   = read_interaction_file(val_path)
    user2test  = read_interaction_file(test_path)

    train_items = {i for v in user2train.values() for i in v}
    val_items   = {i for v in user2val.values()   for i in v}
    test_items  = {i for v in user2test.values()  for i in v}
    all_items   = train_items | val_items | test_items

    logger.info(f"    All items     : {len(all_items):,}")
    logger.info(f"    Train items   : {len(train_items):,}")
    logger.info(f"    Test items    : {len(test_items):,}")

    # Chọn cold items
    cold_items, cold_info = select_true_cold_items(
        test_d    = user2test,
        all_items = all_items,
        ratio     = ratio,
        seed      = seed,
    )

    # Lọc train & val
    def filter_split(d: Dict[int, List[int]]) -> Dict[int, List[int]]:
        result: Dict[int, List[int]] = {}
        for uid, items in d.items():
            kept = [i for i in items if i not in cold_items]
            if kept:
                result[uid] = kept
        return result

    train_cold = filter_split(user2train)
    val_cold   = filter_split(user2val)
    test_cold  = user2test   # giữ nguyên

    # Thống kê
    n_train_before = sum(len(v) for v in user2train.values())
    n_train_after  = sum(len(v) for v in train_cold.values())
    n_val_before   = sum(len(v) for v in user2val.values())
    n_val_after    = sum(len(v) for v in val_cold.values())
    n_test_total   = sum(len(v) for v in user2test.values())
    cold_test_pairs = sum(1 for items in user2test.values() for i in items if i in cold_items)

    logger.info(f"    Train interactions: {n_train_before:,} → {n_train_after:,}")
    logger.info(f"    Val interactions  : {n_val_before:,} → {n_val_after:,}")
    logger.info(f"    Cold test pairs   : {cold_test_pairs:,} ({cold_test_pairs/n_test_total*100:.1f}% của test)")

    # KG
    kg_cold: Optional[List[Tuple[int, int, int]]] = None
    all_triples: List[Tuple[int, int, int]] = []
    if has_kg:
        all_triples = read_kg_file(kg_path)
        kg_cold = filter_kg_by_cold_items(all_triples, cold_items)
        logger.info(f"    KG triples: {len(all_triples):,} → {len(kg_cold):,} ")

    # Lưu file
    out_dir = os.path.join(processed_dir, f"cold_{ratio}")
    os.makedirs(out_dir, exist_ok=True)

    write_interaction_file(os.path.join(out_dir, "train.txt"), train_cold)
    write_interaction_file(os.path.join(out_dir, "val.txt"),   val_cold)
    write_interaction_file(os.path.join(out_dir, "test.txt"),  test_cold)

    with open(os.path.join(out_dir, "cold_items.txt"), "w", encoding="utf-8") as f:
        for iid in sorted(cold_items):
            f.write(f"{iid}\n")

    if kg_cold is not None:
        write_kg_file(os.path.join(out_dir, "kg_full.txt"), kg_cold)

    # Stats
    stats = {
        "dataset": "dataset",
        "ratio": ratio,
        "seed": seed,
        "protocol": "true_cold_start",
        "definition": "Cold items are randomly selected from test items and completely removed from train/val.",
        "eval_metrics": ["HR@10_cold", "NDCG@10_cold", "Recall@20_cold"],
        "n_all_items": len(all_items),
        "n_cold_items": len(cold_items),
        "n_test_items": len(test_items),
        "n_cold_test_pairs": cold_test_pairs,
        # ... (bạn có thể bổ sung thêm nếu cần)
    }

    with open(os.path.join(out_dir, "cold_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"    ✓ Saved to: {out_dir}")


# ------------------------------------------------------------------------------
# CLI (giữ nguyên phần lớn)
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build True Cold-Start splits")
    parser.add_argument("--dataset", choices=["amazon-book", "yelp2018", "all"], default="all")
    parser.add_argument("--ratio", nargs="+", type=int, default=[10, 20, 30])
    parser.add_argument("--data_dir", default="/data/phuongtran/processed")
    parser.add_argument("--seed", type=int, default=COLD_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    datasets = ["amazon-book", "yelp2018"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        processed_dir = os.path.join(args.data_dir, ds)
        if not os.path.isdir(processed_dir):
            logger.error(f"Thư mục không tồn tại: {processed_dir}")
            continue

        for ratio in args.ratio:
            build_cold_split(processed_dir, ds, ratio, args.seed)

    logger.info("True Cold-start split generation hoàn thành.")


if __name__ == "__main__":
    main()
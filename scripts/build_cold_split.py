"""
scripts/build_cold_split.py
------------------------------------------------------------------------------
Cold-start split builder — Long-tail simulation protocol.

ĐỊNH NGHĨA COLD-START (phù hợp với Amazon-Book / LightGCN dataset):
─────────────────────────────────────────────────────────────────────
  Amazon-Book không có "truly cold" items (items chỉ trong test, không
  trong train) vì LightGCN repo đảm bảo mọi test item đều xuất hiện
  trong train (truly_cold = 0 sau khi preprocess).

  Do đó, theo precedent của KGCL (Yang et al., SIGIR 2022) và SimGCL
  (Yu et al., SIGIR 2022), cold-start được mô phỏng bằng long-tail:

      cold_items = bottom X% items theo train interaction frequency
                 = X% items có ÍT lượt tương tác nhất trong training

  Ý nghĩa: model gần như không học được embedding tốt cho các items
  này → mô phỏng tình huống cold / long-tail recommendation.

  Ghi trong paper:
    "Following [KGCL, SimGCL], we simulate cold-start by selecting
     the X% least-interacted items as cold items. Training interactions
     of these items are removed; we evaluate recommendation performance
     on cold items appearing in the test set."

PROTOCOL:
  cold_items  = bottom X% items sorted by train_freq ascending (seed=42)
  Train_cold  = train - {(u,i) | i in cold_items}
  Val_cold    = val   - {(u,i) | i in cold_items}
  Test        = giữ nguyên
  cold_test   = {(u,i) in test | i in cold_items}  ← dùng khi evaluate
  KG_cold     = {(h,r,t) | h in cold_items OR t in cold_items}

3 MỨC:
  Cold-10: bottom 10% least-frequent train items (~9,159 items)
  Cold-20: bottom 20% least-frequent train items (~18,319 items)
  Cold-30: bottom 30% least-frequent train items (~27,479 items)

METRICS:
  HR@10_cold, NDCG@10_cold, Recall@20_cold

SEED: 42 (để Phương tái lập cùng protocol)

Đầu vào (kết quả từ preprocess.py):
  data/processed/<dataset>/train.txt
  data/processed/<dataset>/val.txt
  data/processed/<dataset>/test.txt
  data/processed/<dataset>/kg_full.txt   (chỉ Amazon-Book)

Đầu ra mỗi ratio (10, 20, 30):
  data/processed/<dataset>/cold_<ratio>/train.txt
  data/processed/<dataset>/cold_<ratio>/val.txt
  data/processed/<dataset>/cold_<ratio>/test.txt
  data/processed/<dataset>/cold_<ratio>/kg_full.txt    (chỉ Amazon-Book)
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
# I/O helpers
# ------------------------------------------------------------------------------

def read_interaction_file(path: str) -> Dict[int, List[int]]:
    """Đọc file tương tác: uid item1 item2 ... -> {uid: [iid, ...]}"""
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
    """Ghi file tương tác định dạng LightGCN."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for uid in sorted(user2items.keys()):
            items = user2items[uid]
            if items:
                f.write(f"{uid} " + " ".join(map(str, items)) + "\n")


def read_kg_file(path: str) -> List[Tuple[int, int, int]]:
    """Đọc kg_full.txt định dạng: h<TAB>r<TAB>t."""
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
    """Ghi KG file định dạng: h<TAB>r<TAB>t mỗi dòng, sorted."""
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in sorted(triples):
            f.write(f"{h}\t{r}\t{t}\n")


# ------------------------------------------------------------------------------
# Cold item selection — long-tail simulation
# ------------------------------------------------------------------------------

def select_cold_items_longtail(
    train_d:   Dict[int, List[int]],
    all_items: Set[int],
    ratio:     int,
    seed:      int,
) -> Tuple[Set[int], Dict]:
    """
    Chọn cold items theo long-tail simulation protocol.

    Logic:
      1. Tính train_freq[item] = số lượt tương tác của item trong train
      2. Items không xuất hiện trong train → freq = 0
      3. Sort tất cả items theo freq tăng dần
      4. Lấy bottom (ratio%) làm cold_items
      5. Tie-break tại ranh giới: shuffle nhóm cùng freq bằng seed cố định

    Args:
        train_d:   {uid: [iid, ...]} — training interactions
        all_items: tất cả item IDs (union train+val+test)
        ratio:     10, 20, hoặc 30
        seed:      random seed (42)

    Returns:
        cold_items: Set[int]
        info:       dict thống kê
    """
    # Bước 1: đếm tần suất trong train
    train_freq: Counter = Counter()
    for items in train_d.values():
        for iid in items:
            train_freq[iid] += 1
    # Items không có trong train → freq = 0 (đã tự nhiên là 0 trong Counter)

    n_cold = max(1, int(len(all_items) * ratio / 100))

    # Bước 2: sort theo freq tăng dần, tie-break bằng item_id (deterministic)
    sorted_items = sorted(all_items, key=lambda i: (train_freq.get(i, 0), i))

    # Bước 3: xác định ngưỡng freq tại vị trí n_cold - 1
    cutoff_freq = train_freq.get(sorted_items[n_cold - 1], 0)

    # Bước 4: tách nhóm
    # Items có freq < cutoff → chắc chắn vào cold
    below  = [i for i in sorted_items if train_freq.get(i, 0) <  cutoff_freq]
    # Items có freq == cutoff → lấy một phần để đủ n_cold
    at_cut = [i for i in sorted_items if train_freq.get(i, 0) == cutoff_freq]

    # Bước 5: shuffle nhóm at_cut để tie-break deterministic
    n_needed = n_cold - len(below)
    rng = random.Random(seed)
    rng.shuffle(at_cut)
    selected_from_cut = at_cut[:n_needed]

    cold_items = set(below) | set(selected_from_cut)

    # Thống kê
    freq_values = [train_freq.get(i, 0) for i in cold_items]
    n_zero_freq = sum(1 for f in freq_values if f == 0)

    info = {
        "n_cold":            len(cold_items),
        "n_all_items":       len(all_items),
        "target_fraction":   round(ratio / 100, 4),
        "actual_fraction":   round(len(cold_items) / len(all_items), 4),
        "cutoff_freq":       int(cutoff_freq),
        "cold_freq_min":     int(min(freq_values)) if freq_values else 0,
        "cold_freq_max":     int(max(freq_values)) if freq_values else 0,
        "n_zero_freq_items": n_zero_freq,
        "seed":              seed,
    }

    logger.info(
        f"    Cold items   : {len(cold_items):,} / {len(all_items):,} "
        f"({len(cold_items)/len(all_items):.1%})"
    )
    logger.info(
        f"    Train freq   : min={info['cold_freq_min']}, "
        f"max={info['cold_freq_max']}, cutoff≤{cutoff_freq}"
    )
    logger.info(
        f"    Zero-freq    : {n_zero_freq:,} cold items hoàn toàn "
        f"không có trong train"
    )

    return cold_items, info


# ------------------------------------------------------------------------------
# KG filtering
# ------------------------------------------------------------------------------

def filter_kg_by_cold_items(
    all_triples: List[Tuple[int, int, int]],
    cold_items:  Set[int],
) -> List[Tuple[int, int, int]]:
    """
    Giữ lại triples có ít nhất một đầu là cold item.
    Entity/relation ID KHÔNG bị re-index.
    """
    if not cold_items:
        return []
    return [
        (h, r, t) for h, r, t in all_triples
        if h in cold_items or t in cold_items
    ]


# ------------------------------------------------------------------------------
# Core split builder
# ------------------------------------------------------------------------------

def build_cold_split(
    processed_dir: str,
    dataset:       str,
    ratio:         int,
    seed:          int = COLD_SEED,
) -> None:
    """
    Build một cold split với tỷ lệ cold = ratio%.

    Protocol long-tail:
      1. Đọc train/val/test
      2. Chọn cold_items = bottom ratio% items theo train frequency
      3. Lọc train: xoá interactions với cold_items
      4. Lọc val: xoá interactions với cold_items
      5. Giữ nguyên test (để evaluate cold items)
      6. Lọc KG: giữ triples liên quan cold_items (chỉ Amazon-Book)
      7. Lưu + stats
    """
    logger.info(
        f"\n{'─'*60}\n"
        f"  Building Cold-{ratio} | {dataset} | seed={seed}\n"
        f"  Protocol: bottom {ratio}% least-frequent train items\n"
        f"{'─'*60}"
    )

    # Đường dẫn
    train_path = os.path.join(processed_dir, "train.txt")
    val_path   = os.path.join(processed_dir, "val.txt")
    test_path  = os.path.join(processed_dir, "test.txt")
    kg_path    = os.path.join(processed_dir, "kg_full.txt")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Không tìm thấy: {p}. Chạy preprocess.py trước."
            )

    has_kg = (dataset in DATASETS_KG) and os.path.exists(kg_path)

    # Đọc data
    user2train = read_interaction_file(train_path)
    user2val   = read_interaction_file(val_path)
    user2test  = read_interaction_file(test_path)

    train_items: Set[int] = {i for v in user2train.values() for i in v}
    val_items:   Set[int] = {i for v in user2val.values()   for i in v}
    test_items:  Set[int] = {i for v in user2test.values()  for i in v}
    all_items = train_items | val_items | test_items

    logger.info(f"    All items  : {len(all_items):,}")
    logger.info(f"    Train items: {len(train_items):,}")
    logger.info(f"    Val items  : {len(val_items):,}")
    logger.info(f"    Test items : {len(test_items):,}")
    logger.info(
        f"    Truly cold (test-train-val): "
        f"{len(test_items - train_items - val_items):,}  "
        f"[Amazon-Book = 0, long-tail protocol used]"
    )

    # Chọn cold items
    cold_items, cold_info = select_cold_items_longtail(
        train_d   = user2train,
        all_items = all_items,
        ratio     = ratio,
        seed      = seed,
    )

    # Overlap stats
    cold_in_test  = cold_items & test_items
    cold_in_train = cold_items & train_items
    cold_in_val   = cold_items & val_items

    logger.info(f"    Cold ∩ test : {len(cold_in_test):,}  (có ground truth để evaluate)")
    logger.info(f"    Cold ∩ train: {len(cold_in_train):,}  (sẽ bị xoá khỏi train)")
    logger.info(f"    Cold ∩ val  : {len(cold_in_val):,}   (sẽ bị xoá khỏi val)")

    if len(cold_in_test) == 0:
        logger.warning(
            "    ⚠ Không có cold items nào xuất hiện trong test! "
            "Cold evaluation sẽ rỗng."
        )

    # Lọc train và val — xoá cold items
    def filter_split(d: Dict[int, List[int]]) -> Dict[int, List[int]]:
        result: Dict[int, List[int]] = {}
        for uid, items in d.items():
            kept = [i for i in items if i not in cold_items]
            if kept:
                result[uid] = kept
        return result

    train_cold = filter_split(user2train)
    val_cold   = filter_split(user2val)
    test_cold  = user2test  # giữ nguyên

    # Thống kê interactions
    n_train_before = sum(len(v) for v in user2train.values())
    n_train_after  = sum(len(v) for v in train_cold.values())
    n_val_before   = sum(len(v) for v in user2val.values())
    n_val_after    = sum(len(v) for v in val_cold.values())
    n_test_total   = sum(len(v) for v in user2test.values())
    cold_test_pairs = sum(
        1 for items in user2test.values() for i in items if i in cold_items
    )

    logger.info(
        f"    Train: {n_train_before:,} → {n_train_after:,} "
        f"(-{n_train_before - n_train_after:,} interactions)"
    )
    logger.info(
        f"    Val  : {n_val_before:,} → {n_val_after:,} "
        f"(-{n_val_before - n_val_after:,} interactions)"
    )
    logger.info(
        f"    Cold test pairs: {cold_test_pairs:,} "
        f"({cold_test_pairs/n_test_total*100:.1f}% của test)"
    )

    # KG
    kg_cold: Optional[List[Tuple[int, int, int]]] = None
    all_triples: List[Tuple[int, int, int]] = []
    if has_kg:
        all_triples = read_kg_file(kg_path)
        kg_cold = filter_kg_by_cold_items(all_triples, cold_items)
        logger.info(
            f"    KG: {len(all_triples):,} → {len(kg_cold):,} triples "
            f"({len(kg_cold)/len(all_triples)*100:.1f}%)"
        )

    # Lưu files
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

    # cold_stats.json — đầy đủ để Phương tái lập
    stats = {
        "dataset":   dataset,
        "ratio":     ratio,
        "seed":      seed,

        # Protocol
        "protocol":  "long-tail cold-start simulation",
        "definition": (
            f"cold_items = bottom {ratio}% items by train interaction frequency "
            f"(least-interacted items). "
            "Amazon-Book has 0 truly-cold items (all test items appear in train), "
            "so long-tail simulation is used following KGCL/SimGCL."
        ),
        "paper_note": (
            "Following KGCL (Yang et al., SIGIR 2022) and SimGCL (Yu et al., SIGIR 2022), "
            f"we simulate cold-start by selecting the {ratio}% least-interacted items "
            "as cold items. Training/validation interactions of these items are removed. "
            "We evaluate on cold items appearing in the test set."
        ),
        "reproduce_cmd": (
            f"python scripts/build_cold_split.py "
            f"--dataset {dataset} --ratio {ratio} --seed {seed} "
            f"--data_dir <your_data_dir>"
        ),

        # Item statistics
        "n_all_items":           len(all_items),
        "n_cold_items":          len(cold_items),
        "target_fraction":       cold_info["target_fraction"],
        "actual_fraction":       cold_info["actual_fraction"],
        "cutoff_train_freq":     cold_info["cutoff_freq"],
        "cold_freq_min":         cold_info["cold_freq_min"],
        "cold_freq_max":         cold_info["cold_freq_max"],
        "n_zero_freq_items":     cold_info["n_zero_freq_items"],

        # Overlap
        "n_cold_in_test":        len(cold_in_test),
        "n_cold_in_train":       len(cold_in_train),
        "n_cold_in_val":         len(cold_in_val),

        # Interactions
        "n_train_before":        n_train_before,
        "n_train_after":         n_train_after,
        "n_train_removed":       n_train_before - n_train_after,
        "n_val_before":          n_val_before,
        "n_val_after":           n_val_after,
        "n_val_removed":         n_val_before - n_val_after,
        "n_test_total":          n_test_total,
        "n_cold_test_pairs":     cold_test_pairs,
        "cold_test_pairs_pct":   round(cold_test_pairs / n_test_total * 100, 2)
                                 if n_test_total else 0.0,

        # KG (Amazon-Book only)
        "has_kg":                has_kg,
        "n_kg_triples_original": len(all_triples) if all_triples else None,
        "n_kg_triples_cold":     len(kg_cold) if kg_cold is not None else None,
        "kg_cold_pct":           round(len(kg_cold) / len(all_triples) * 100, 2)
                                 if kg_cold and all_triples else None,

        # Metrics to report
        "eval_metrics": ["hr@10_cold", "ndcg@10_cold", "recall@20_cold"],
    }

    with open(os.path.join(out_dir, "cold_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"    ✓ Saved to: {out_dir}")
    logger.info(
        f"    Files: train.txt | val.txt | test.txt | "
        f"cold_items.txt | cold_stats.json"
        + (" | kg_full.txt" if kg_cold is not None else "")
    )


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build cold-start splits — long-tail simulation.\n"
            "cold_items = bottom X%% items by train interaction frequency.\n"
            "Amazon-Book: truly_cold=0 → long-tail is the only option.\n"
            "seed=42 fixed for reproducibility."
        )
    )
    parser.add_argument(
        "--dataset", choices=["amazon-book", "yelp2018", "all"], default="all"
    )
    parser.add_argument(
        "--ratio", nargs="+", type=int, default=[10, 20, 30],
        help="Target %% cold items. Default: 10 20 30"
    )
    parser.add_argument("--data_dir", default="/data/phuongtran/processed")
    parser.add_argument("--seed", type=int, default=COLD_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    datasets = (
        ["amazon-book", "yelp2018"] if args.dataset == "all" else [args.dataset]
    )

    for ds in datasets:
        processed_dir = os.path.join(args.data_dir, ds)
        if not os.path.isdir(processed_dir):
            logger.error(
                f"Thư mục không tồn tại: {processed_dir}. "
                "Chạy preprocess.py trước."
            )
            continue

        logger.info("=" * 65)
        logger.info(f"Dataset   : {ds}")
        logger.info(f"Protocol  : long-tail cold-start simulation")
        logger.info(f"Seed      : {args.seed}")
        logger.info(f"Ratios    : {args.ratio}")
        logger.info("=" * 65)

        for ratio in args.ratio:
            build_cold_split(
                processed_dir=processed_dir,
                dataset=ds,
                ratio=ratio,
                seed=args.seed,
            )

    logger.info("\n" + "=" * 65)
    logger.info("Cold split generation hoàn thành.")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
"""
scripts/build_cold_split.py
------------------------------------------------------------------------------
Cold-start split builder.

Protocol:
  cold_items = X% items duoc sample tu TAT CA items, uu tien:
    - Items chi xuat hien trong test (cold items thuc su)
    - Neu khong du: items xuat hien trong test voi it luot train nhat

  Train_cold = {(u,i) in train | i not in cold_items}
  Val_cold   = {(u,i) in val   | i not in cold_items}
  Test       = giu nguyen (cold items xuat hien o day de evaluate)

  cold_test  = {(u,i) in test | i in cold_items}  <- dung khi evaluate cold

  KG_cold    = {(h,r,t) in kg_full | h in cold_items OR t in cold_items}
               Giu nguyen entity/relation ID -- khong re-index.

Luu y:
  - Trong LightGCN dataset, hau het items xuat hien ca trong train lan test
    (khong co items "purely cold"). Do do, ta relaxe constraint:
    cold_items = items it xuat hien nhat trong train (bottom X%)
    -> Mo phong kich ban long-tail / it tuong tac
  - KG chi build cho Amazon-Book (Yelp2018 la CF only).
  - Entity/relation ID trong kg_full.txt KHONG bi re-index de dam bao
    load duoc cung model checkpoint.

Dau vao (ket qua tu preprocess.py):
  data/processed/<dataset>/train.txt
  data/processed/<dataset>/val.txt
  data/processed/<dataset>/test.txt
  data/processed/<dataset>/kg_full.txt   (chi Amazon-Book)

Dau ra moi ratio (10, 20, 30):
  data/processed/<dataset>/cold_<ratio>/train.txt
  data/processed/<dataset>/cold_<ratio>/val.txt
  data/processed/<dataset>/cold_<ratio>/test.txt
  data/processed/<dataset>/cold_<ratio>/kg_full.txt   (chi Amazon-Book)
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

COLD_SEED    = 42
DATASETS_KG  = {"amazon-book"}   # dataset co KG


# ------------------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------------------

def read_interaction_file(path: str) -> Dict[int, List[int]]:
    """Doc file tuong tac: uid item1 item2 ... -> {uid: [iid, ...]}"""
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
    """Ghi file tuong tac dinh dang LightGCN."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for uid in sorted(user2items.keys()):
            items = user2items[uid]
            if items:
                f.write(f"{uid} " + " ".join(map(str, items)) + "\n")


def read_kg_file(path: str) -> List[Tuple[int, int, int]]:
    """
    Doc kg_full.txt dinh dang: h<TAB>r<TAB>t moi dong.
    Tra ve list of (h, r, t).
    """
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
    """Ghi KG file dinh dang: h<TAB>r<TAB>t moi dong, sorted."""
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in sorted(triples):
            f.write(f"{h}\t{r}\t{t}\n")


# ------------------------------------------------------------------------------
# Cold item selection
# ------------------------------------------------------------------------------

def select_cold_items(
    train_items: Set[int],
    val_items:   Set[int],
    test_items:  Set[int],
    all_items:   Set[int],
    ratio:       int,
    seed:        int,
) -> Set[int]:
    """
    Chon X% cold items tu all_items theo thu tu uu tien:

    Uu tien 1: items trong test nhung KHONG trong train va val (truly cold).
    Uu tien 2: items trong test VA trong train, sort theo train freq tang dan
               (it tuong tac nhat truoc).
    Uu tien 3: items chi trong train, sort theo train freq tang dan.

    Trong moi nhom, shuffle nua dau (freq thap nhat) de tranh deterministic
    bias trong truong hop nhieu items co cung tan suat.
    """
    train_freq = Counter(
        i for items_list in [train_items] for i in items_list
    )
    # train_freq can tinh dung tu interactions, khong phai set
    # -- caller se truyen train_freq tu ben ngoai neu can chinh xac hon.
    # O day dung set lam proxy: moi item co count >= 1.
    # Caller truyen train_freq rieng qua argument de chinh xac.
    # (giu signature don gian; tinh lai tu train_d o ham goi)

    n_cold = max(1, int(len(all_items) * ratio / 100))

    truly_cold      = list(test_items - train_items - val_items)
    in_test_train   = sorted(test_items & train_items,
                             key=lambda i: train_freq.get(i, 0))
    train_only      = sorted(train_items - test_items,
                             key=lambda i: train_freq.get(i, 0))

    rng = random.Random(seed)
    rng.shuffle(truly_cold)
    # Shuffle nua dau (least frequent) cua in_test_train de bo sung diversity
    half = max(1, len(in_test_train) // 2)
    shuffled_half = in_test_train[:half]
    rng.shuffle(shuffled_half)
    in_test_train = shuffled_half + in_test_train[half:]

    candidate_pool = truly_cold + in_test_train + train_only

    # Dedup giu thu tu
    seen: Set[int] = set()
    ordered: List[int] = []
    for i in candidate_pool:
        if i not in seen:
            seen.add(i)
            ordered.append(i)

    return set(ordered[:n_cold])


def select_cold_items_with_freq(
    train_d:   Dict[int, List[int]],
    val_items: Set[int],
    test_items: Set[int],
    all_items:  Set[int],
    ratio:      int,
    seed:       int,
) -> Set[int]:
    """
    Wrapper chinh xac: tinh train_freq tu interaction dict (dem so users,
    khong phai so occurrences raw) truoc khi goi select_cold_items.
    """
    train_freq: Counter = Counter()
    train_items: Set[int] = set()
    for items in train_d.values():
        for i in items:
            train_freq[i] += 1
            train_items.add(i)

    n_cold = max(1, int(len(all_items) * ratio / 100))

    val_set  = val_items
    test_set = test_items

    truly_cold    = list(test_set - train_items - val_set)
    in_test_train = sorted(test_set & train_items,
                           key=lambda i: train_freq.get(i, 0))
    train_only    = sorted(train_items - test_set,
                           key=lambda i: train_freq.get(i, 0))

    rng = random.Random(seed)
    rng.shuffle(truly_cold)
    half = max(1, len(in_test_train) // 2)
    shuffled_half = in_test_train[:half]
    rng.shuffle(shuffled_half)
    in_test_train = shuffled_half + in_test_train[half:]

    candidate_pool = truly_cold + in_test_train + train_only
    seen: Set[int] = set()
    ordered: List[int] = []
    for i in candidate_pool:
        if i not in seen:
            seen.add(i)
            ordered.append(i)

    return set(ordered[:n_cold])


# ------------------------------------------------------------------------------
# KG filtering
# ------------------------------------------------------------------------------

def filter_kg_by_cold_items(
    all_triples: List[Tuple[int, int, int]],
    cold_items:  Set[int],
) -> List[Tuple[int, int, int]]:
    """
    Giu lai cac triple (h, r, t) ma h hoac t la cold item.

    Nguyen tac:
      - Item-item triples (rel 0,1,4,5): giu neu it nhat mot dau la cold item.
        Dieu nay dam bao model thay duoc co-occurrence pattern cua cold items,
        kể ca khi partner item khong phai cold.
      - Item-category / item-brand triples (rel 2,3,6,7): giu neu item (dau
        hay cuoi tuy chieu) la cold item.
      - Trong thuc te ca hai loai deu duoc xu ly bang cung mot dieu kien:
        h in cold_items OR t in cold_items.

    Entity/relation ID KHONG bi re-index.
    Ket qua co the load truc tiep bang cung entity embedding matrix.

    So sanh voi strict mode (ca hai dau deu cold):
      - Strict: KG nho hon, chi phan anh cold-cold relationships.
      - Loose (dung o day): KG lon hon, cold items van duoc ket noi voi warm
        items -> phong phu hon ve thong tin KG khi evaluate.
    """
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
    Build mot cold split voi ty le cold = ratio%.

    Quy trinh:
      1. Doc train/val/test
      2. Chon cold_items (X% items it train nhat, uu tien items co trong test)
      3. Loc train/val: bo cold items
      4. Giu nguyen test (cold items van o day de evaluate)
      5. [Neu Amazon-Book] Loc kg_full.txt: giu triple lien quan cold items
      6. Luu tat ca + stats
    """
    logger.info(f"  Building Cold-{ratio} split for {dataset}  (seed={seed}) ...")

    # -- Doc dau vao ----------------------------------------------------------
    train_path = os.path.join(processed_dir, "train.txt")
    val_path   = os.path.join(processed_dir, "val.txt")
    test_path  = os.path.join(processed_dir, "test.txt")
    kg_path    = os.path.join(processed_dir, "kg_full.txt")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Khong tim thay: {p}. Chay preprocess.py truoc."
            )

    has_kg = dataset in DATASETS_KG and os.path.exists(kg_path)
    if dataset in DATASETS_KG and not os.path.exists(kg_path):
        logger.warning(
            f"  kg_full.txt khong tim thay tai {kg_path}. "
            "Se bo qua buoc build KG cold split."
        )

    user2train = read_interaction_file(train_path)
    user2val   = read_interaction_file(val_path)
    user2test  = read_interaction_file(test_path)

    train_items: Set[int] = {i for v in user2train.values() for i in v}
    val_items:   Set[int] = {i for v in user2val.values()   for i in v}
    test_items:  Set[int] = {i for v in user2test.values()  for i in v}
    all_items = train_items | val_items | test_items

    # -- Chon cold items ------------------------------------------------------
    cold_items = select_cold_items_with_freq(
        train_d    = user2train,
        val_items  = val_items,
        test_items = test_items,
        all_items  = all_items,
        ratio      = ratio,
        seed       = seed,
    )

    cold_in_test  = cold_items & test_items
    cold_in_train = cold_items & train_items
    cold_in_val   = cold_items & val_items

    logger.info(
        f"    Cold items  : {len(cold_items):,} / {len(all_items):,} "
        f"({len(cold_items) / len(all_items):.1%})"
    )
    logger.info(f"    In test     : {len(cold_in_test):,}  (se duoc evaluate)")
    logger.info(f"    In train    : {len(cold_in_train):,}  (se bi xoa khoi train)")
    logger.info(f"    In val      : {len(cold_in_val):,}   (se bi xoa khoi val)")

    if len(cold_in_test) == 0:
        logger.warning(
            "    0 cold items xuat hien trong test. "
            "Cold evaluation se rong. "
            "Kiem tra lai dataset -- co the tat ca test items deu da co trong train."
        )

    # -- Loc train / val ------------------------------------------------------
    def filter_split(d: Dict[int, List[int]]) -> Dict[int, List[int]]:
        result: Dict[int, List[int]] = {}
        for uid, items in d.items():
            kept = [i for i in items if i not in cold_items]
            if kept:
                result[uid] = kept
        return result

    train_cold = filter_split(user2train)
    val_cold   = filter_split(user2val)
    test_cold  = user2test   # Test KHONG thay doi

    # -- Loc KG ---------------------------------------------------------------
    kg_cold: Optional[List[Tuple[int, int, int]]] = None
    if has_kg:
        logger.info(f"    Doc KG tu: {kg_path} ...")
        all_triples = read_kg_file(kg_path)
        logger.info(f"    KG goc: {len(all_triples):,} triples")
        kg_cold = filter_kg_by_cold_items(all_triples, cold_items)
        logger.info(f"    KG cold: {len(kg_cold):,} triples "
                    f"({len(kg_cold) / len(all_triples) * 100:.1f}% cua KG goc)")

    # -- Luu ------------------------------------------------------------------
    out_dir = os.path.join(processed_dir, f"cold_{ratio}")
    os.makedirs(out_dir, exist_ok=True)

    write_interaction_file(os.path.join(out_dir, "train.txt"), train_cold)
    write_interaction_file(os.path.join(out_dir, "val.txt"),   val_cold)
    write_interaction_file(os.path.join(out_dir, "test.txt"),  test_cold)

    if kg_cold is not None:
        write_kg_file(os.path.join(out_dir, "kg_full.txt"), kg_cold)
        logger.info(f"    kg_full.txt da luu: {len(kg_cold):,} triples")

    with open(os.path.join(out_dir, "cold_items.txt"), "w", encoding="utf-8") as f:
        for iid in sorted(cold_items):
            f.write(f"{iid}\n")

    # -- Thong ke -------------------------------------------------------------
    n_train_after = sum(len(v) for v in train_cold.values())
    n_val_after   = sum(len(v) for v in val_cold.values())
    n_test_total  = sum(len(v) for v in test_cold.values())
    n_test_train_before = sum(len(v) for v in user2train.values())
    n_val_before        = sum(len(v) for v in user2val.values())

    cold_test_pairs = sum(
        1 for items in test_cold.values() for i in items if i in cold_items
    )

    stats = {
        "dataset":                       dataset,
        "ratio":                         ratio,
        "seed":                          seed,
        "strategy":                      "least_frequent_train_items_preferring_test_items",

        # Cold items
        "n_cold_items":                  len(cold_items),
        "n_all_items":                   len(all_items),
        "cold_fraction":                 round(len(cold_items) / len(all_items), 4),
        "n_cold_in_test":                len(cold_in_test),
        "n_cold_in_train":               len(cold_in_train),
        "n_cold_in_val":                 len(cold_in_val),

        # Train / val sau khi loc
        "n_train_before":                n_test_train_before,
        "n_train_after":                 n_train_after,
        "n_train_removed":               n_test_train_before - n_train_after,
        "n_val_before":                  n_val_before,
        "n_val_after":                   n_val_after,
        "n_val_removed":                 n_val_before - n_val_after,

        # Test (khong thay doi, nhung log cold pairs)
        "n_test_total":                  n_test_total,
        "n_cold_test_pairs":             cold_test_pairs,
        "cold_test_pairs_pct":           round(cold_test_pairs / n_test_total * 100, 2)
                                         if n_test_total else 0.0,

        # KG (chi Amazon-Book)
        "has_kg":                        has_kg,
        "n_kg_triples_original":         len(all_triples) if kg_cold is not None else None,
        "n_kg_triples_cold":             len(kg_cold)     if kg_cold is not None else None,
        "kg_triples_cold_pct":           round(len(kg_cold) / len(all_triples) * 100, 2)
                                         if kg_cold and all_triples else None,

        "note": (
            f"Train/val: bo cac interaction voi cold items (bottom {ratio}% least freq). "
            "Test: giu nguyen. "
            "KG: giu triple co it nhat mot dau la cold item. "
            "Entity/relation ID khong bi re-index."
        ),
    }

    with open(os.path.join(out_dir, "cold_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"    Cold test pairs (evaluate): {cold_test_pairs:,}")
    logger.info(f"    Saved to: {out_dir}")
    logger.info(f"    Files: train.txt | val.txt | test.txt | cold_items.txt | cold_stats.json"
                + (" | kg_full.txt" if kg_cold is not None else ""))


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cold-start splits (train/val/test + kg_full.txt)"
    )
    parser.add_argument(
        "--dataset", choices=["amazon-book", "yelp2018", "all"], default="all"
    )
    parser.add_argument(
        "--ratio", nargs="+", type=int, default=[10, 20, 30],
        help="Phan tram cold items. Mac dinh: 10 20 30"
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
                f"Thu muc khong ton tai: {processed_dir}. "
                "Chay preprocess.py truoc."
            )
            continue

        logger.info("=" * 65)
        logger.info(f"Dataset: {ds}")
        logger.info("=" * 65)

        for ratio in args.ratio:
            build_cold_split(
                processed_dir = processed_dir,
                dataset       = ds,
                ratio         = ratio,
                seed          = args.seed,
            )

    logger.info("=" * 65)
    logger.info("Cold split generation hoan thanh.")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
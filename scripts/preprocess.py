"""
scripts/preprocess.py
------------------------------------------------------------------------------
Preprocessing pipeline cho Amazon-Book (LightGCN repo + meta_Books.json.gz)
va Yelp2018 (SimGCL / QRec repo).

==============================================================================
NGUON DU LIEU AMAZON-BOOK
==============================================================================
CF interactions  -- LightGCN-PyTorch repo:
  https://github.com/gusye1234/LightGCN-PyTorch/tree/master/data/amazon-book
  Files: train.txt, test.txt, item_list.txt, user_list.txt

KG (tu xay dung) -- Amazon product metadata:
  meta_Books.json.gz  (McAuley Amazon dataset)
  Dung: item_list.txt de anh xa ASIN -> item_id
        meta_Books.json.gz de trich xuat thuoc tinh:
          - also_buy   -> relation 0 (also_bought)
          - also_view  -> relation 1 (also_viewed)
          - categories -> relation 2 (belongs_to_category)
          - brand/publisher -> relation 3 (has_brand)
            Trich xuat theo thu tu uu tien:
              1. Truong "brand" (neu co)
              2. Truong "publisher" (neu co)
              3. Truong "details" -> key "Publisher" / "publisher" (HTML-embedded)
              4. Leaf category cua path dau tien lam fallback

==============================================================================
FORMAT FILE LIGHTGCN REPO
==============================================================================
train.txt / test.txt:
  user_id item1 item2 item3 ...   (user-item list moi dong, da indexed)

item_list.txt:
  org_id  remap_id
  0060090383  0
  (KHONG co header "item_id  org_id" trong mot so phien ban -- code tu detect)

user_list.txt:
  org_id  remap_id

==============================================================================
PROTOCOL CHUAN (de ket qua so sanh duoc voi paper)
==============================================================================
  - Giu nguyen train.txt goc -> train pool
  - Giu nguyen test.txt goc  -> test set (KHONG thay doi)
  - Val = 10% tu train pool, user-wise:
      * Neu co timestamp: sort theo time, lay 10% cuoi lam val
      * Neu khong co timestamp (LightGCN repo): shuffle(seed=42), lay 10% cuoi
  - 5-core filtering tren train+test truoc khi split val

Pipeline:
  1. Doc train.txt, test.txt
  2. 5-core filtering (tren union train+test, sau khi dedup)
  3. ID remapping user->[0,N), item->[0,M) -- dung union de nhat quan
  4. Tach val tu train pool (10%, user-wise, seed=42, lay 10% cuoi sau shuffle)
  5. Xay dung KG tu meta_Books.json.gz + item_list.txt
     -- ghi ca forward relation (0-3) lan inverse relation (4-7)
  6. Tinh thong ke (coverage tinh tu so item thuc su co metadata)
  7. Kiem tra reproducibility bang fingerprint ket qua (khong re-run pipeline)
  8. Luu tat ca files

==============================================================================
FIXES SO VOI PHIEN BAN CU
==============================================================================
[FIX-1] read_item_list_lightgcn: doc dung cot (org_id=parts[0], remap_id=parts[1]).
        Tra ve {remap_id -> ASIN} -- day la ID goc cua LightGCN repo, dung truc tiep
        lam key tra cuu trong item_map (old->new sau 5-core + remap).

[FIX-2] coverage: tinh tu n_matched / n_items thay vi len(item2entity)/n_items
        (item2entity luon = n_items -> coverage luon = 1.0 -- sai hoan toan).

[FIX-3] split_val_from_train: lay 10% CUOI (arr[-n_val:]) lam val sau shuffle,
        phu hop voi ca nhanh "no-timestamp" lan semantic "held-out last interactions".

[FIX-4] five_core_filter: dedup all_pairs truoc khi dem de tranh count thua do
        duplicate trong file goc.

[FIX-5] Them inverse relations (rel_id + N_RELATIONS/2) vao KG output.
        n_relations = 8 (4 forward + 4 inverse), khop voi benchmark papers.

[FIX-6] verify_reproducibility: dung fingerprint cua ket qua da tinh (khong re-run
        pipeline 3 lan -- tiet kiem 3x I/O voi file ~3M interactions).

[FIX-7] flat_cats fallback: dung leaf cua path DAU TIEN (path[0][-1]) thay vi
        flat_cats[-1] (phan tu cuoi cua toan bo list da flatten -- khong nhat quan).

[FIX-8] import ast, import re: chuyen len top-level, ra khoi vong lap.

[FIX-9] Bo asin2entity tra ra ngoai (khong dung o dau); giu lai trong ham de log.

[FIX-10] write_interaction_file: bo os.makedirs ben trong -- goi mot lan o ngoai.

[FIX-11] pairs_to_dict: bo tham so dedup vo nghia (ca hai nhanh if/else deu dung
         set.add() -> dedup=False khong co tac dung). Su dung list de giu duplicate
         khi dedup=False, set khi dedup=True.

[FIX-12] compute_stats: bo sung density_train (chi tinh tren train) va dat ten lai
         density -> density_all_splits de tranh nham lan khi so sanh voi paper.
         Bo sung n_items_without_kg va kg_coverage_note cho Amazon-Book.

[FIX-13] Sua comment build_kg_from_meta: brand entity ID duoc assign ngay trong
         vong lap chinh (khong phai pass 2), brand_pending chi defer viec tao triple.
"""

import argparse
import ast
import gzip
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_script_logger
from utils.seed import set_seed

logger = get_script_logger("preprocess")

# So relations forward; inverse = forward + N_FORWARD_RELATIONS
N_FORWARD_RELATIONS = 4


# ------------------------------------------------------------------------------
# I/O helpers -- doc LightGCN format
# ------------------------------------------------------------------------------

def read_lightgcn_interaction(path: str) -> List[Tuple[int, int]]:
    """
    Doc LightGCN format: moi dong  user_id  item1  item2  ...
    Bo qua dong khong parse duoc thanh so.
    """
    pairs: List[Tuple[int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                uid = int(parts[0])
                for tok in parts[1:]:
                    pairs.append((uid, int(tok)))
            except ValueError:
                continue
    return pairs


def read_item_list_lightgcn(path: str) -> Dict[int, str]:
    """
    Doc item_list.txt cua LightGCN/KGAT repo.

    Format thuc te (khong nhat thiet co header):
        org_id  remap_id
        0060090383  0
        0374157065  1
        ...

    [FIX-1] Tra ve {remap_id(int) -> asin(str)}.
    remap_id la item_id duoc dung trong train.txt/test.txt cua LightGCN repo,
    do do day la key phu hop de tra cuu trong item_map (old_item_id -> new_item_id)
    sau buoc 5-core filter + remap.

    Tu dong bo qua dong header neu parts[1] khong parse duoc thanh int.
    """
    remap2asin: Dict[int, str] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            asin = parts[0].strip()
            try:
                remap_id = int(parts[1])
            except ValueError:
                # Dong header hoac dong khong hop le -- bo qua
                continue
            remap2asin[remap_id] = asin

    return remap2asin


def pairs_to_dict(
    pairs: List[Tuple[int, int]],
    dedup: bool = True,
) -> Dict[int, List[int]]:
    """
    (uid, iid) list -> {uid: sorted [iid, ...]}

    [FIX-11] Sua lai logic dedup:
      - dedup=True  (mac dinh): dung set, loai bo interaction trung lap.
        Dung cho toan bo pipeline hien tai.
      - dedup=False: giu nguyen duplicate bang list (khong dung set).
        Huu ich neu caller can biet tan suat tuong tac chinh xac.
    """
    if dedup:
        d: Dict[int, Set[int]] = defaultdict(set)
        for uid, iid in pairs:
            d[uid].add(iid)
        return {u: sorted(v) for u, v in d.items()}
    else:
        dl: Dict[int, List[int]] = defaultdict(list)
        for uid, iid in pairs:
            dl[uid].append(iid)
        return {u: sorted(v) for u, v in dl.items()}


def write_interaction_file(path: str, user2items: Dict[int, List[int]]) -> None:
    """
    Ghi file tuong tac. Thu muc phai duoc tao truoc boi caller.
    [FIX-10] Bo os.makedirs ben trong -- tranh goi thua cho moi file.
    """
    with open(path, "w", encoding="utf-8") as f:
        for uid in sorted(user2items.keys()):
            items = user2items[uid]
            if items:
                f.write(f"{uid} " + " ".join(map(str, items)) + "\n")


def write_kg_file(path: str, triples: List[Tuple[int, int, int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in sorted(triples):
            f.write(f"{h}\t{r}\t{t}\n")


# ------------------------------------------------------------------------------
# KG builder tu meta_Books.json.gz
# ------------------------------------------------------------------------------

def _iter_meta_books(gz_path: str) -> Iterator[dict]:
    """
    Generator doc tung dong JSON tu meta_Books.json.gz.
    Moi dong la mot JSON object (metadata cua 1 san pham).
    Bo qua dong khong parse duoc.
    """
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                try:
                    # Fallback: mot so dong dung Python dict literal (single-quote)
                    yield ast.literal_eval(line)
                except Exception:
                    continue


def _extract_brand(record: dict, first_category_path: List[str]) -> str:
    """
    Trich xuat brand/publisher theo thu tu uu tien:
      1. Truong "brand"
      2. Truong "publisher"
      3. Truong "details" -> key Publisher/publisher/Label/Studio
      4. Leaf cua category path DAU TIEN (first_category_path[-1])

    [FIX-7] Fallback dung leaf cua path dau tien thay vi flat_cats[-1],
    dam bao nhat quan khi item co nhieu category path.

    [FIX-8] re duoc import o top-level.
    """
    # 1. "brand"
    raw = record.get("brand", "")
    if isinstance(raw, list):
        raw = " ".join(str(x) for x in raw)
    val = str(raw).strip()
    if val:
        return val

    # 2. "publisher"
    raw = record.get("publisher", "")
    if isinstance(raw, list):
        raw = " ".join(str(x) for x in raw)
    val = str(raw).strip()
    if val:
        return val

    # 3. "details"
    details = record.get("details", {})
    if isinstance(details, dict):
        for key in ("Publisher", "publisher", "Label", "Studio"):
            v = details.get(key, "")
            if v and str(v).strip():
                return str(v).strip()
    elif isinstance(details, str) and details.strip():
        m = re.search(
            r'"(?:Publisher|publisher|Label|Studio)"\s*:\s*"([^"]+)"',
            details,
        )
        if m:
            return m.group(1).strip()

    # 4. Fallback: leaf cua path DAU TIEN
    if first_category_path:
        return first_category_path[-1]

    return ""


def build_kg_from_meta(
    gz_path:   str,
    item2asin: Dict[int, str],   # {remap_id -> ASIN}  [FIX-1]
    item_map:  Dict[int, int],   # {old_item_id -> new_item_id} sau remap
    n_items:   int,
) -> Tuple[
    List[Tuple[int, int, int]],  # triples (forward + inverse)
    int,                          # n_entities
    int,                          # n_relations (= 2 x N_FORWARD_RELATIONS)
    float,                        # kg_coverage (thuc su co metadata)
]:
    """
    Build KG from meta_Books.json.gz, bao gom ca inverse relations.

    Entity space:
      [0, n_items)                 : item entities
      [n_items, n_items+C)         : category entities
      [n_items+C, n_items+C+P)     : publisher/brand entities

    Forward relations  (0-3):
      0 : also_bought   (item->item)
      1 : also_viewed   (item->item)
      2 : has_category  (item->category)
      3 : has_brand     (item->brand/publisher/leaf_category)

    Inverse relations  (4-7):   rel_inv = rel_fwd + N_FORWARD_RELATIONS
      4 : also_bought_by   (item<-item)
      5 : also_viewed_by   (item<-item)
      6 : category_of      (category<-item)
      7 : brand_of         (brand<-item)

    [FIX-5] Them inverse relations -- n_relations = 8, khop voi benchmark papers
            (KGAT, KGCL, RippleNet thuong dung bidirectional KG).
    """
    logger.info(f"  [KG] Xay dung KG tu: {gz_path}")
    logger.info(f"       Items can map: {n_items:,}")

    # --- ASIN -> new_item_id -------------------------------------------------
    # item2asin: {remap_id -> ASIN}
    # item_map:  {remap_id -> new_item_id}
    # [FIX-1] Dung item_map.get(remap_id) -- nhat quan voi cach build item_map
    asin2new_item: Dict[str, int] = {}
    for remap_id, asin in item2asin.items():
        new_iid = item_map.get(remap_id)
        if new_iid is not None:
            asin2new_item[asin] = new_iid

    n_mapped = len(asin2new_item)
    logger.info(f"       ASIN mapped: {n_mapped:,}/{n_items:,}")

    # --- Entity containers ---------------------------------------------------
    category2eid: Dict[str, int] = {}
    brand2eid:    Dict[str, int] = {}
    next_eid = n_items

    # Forward triples (dedup bang set)
    fwd_triples: Set[Tuple[int, int, int]] = set()

    # [FIX-13] brand entity ID duoc assign ngay trong vong lap chinh;
    # brand_pending chi defer viec tao triple (khong phai "pass 2 assign").
    brand_pending: List[Tuple[int, str]] = []

    REL_ALSO_BOUGHT = 0
    REL_ALSO_VIEWED = 1
    REL_CATEGORY    = 2
    REL_BRAND       = 3

    n_processed = 0
    n_matched   = 0

    logger.info("       Dang doc metadata ...")

    for record in _iter_meta_books(gz_path):
        n_processed += 1
        if n_processed % 500_000 == 0:
            logger.info(
                f"       ... {n_processed:,} records | "
                f"{n_matched:,} matched | "
                f"{len(fwd_triples):,} forward triples"
            )

        asin = str(record.get("asin", "")).strip()
        if not asin:
            continue

        src_item = asin2new_item.get(asin)
        if src_item is None:
            continue

        n_matched += 1

        # -- related (also_bought / also_viewed) ------------------------------
        # McAuley dataset ton tai 2 format:
        #   - 2018 version: 'related' -> {'also_bought': [...], 'also_viewed': [...]}
        #   - 2023 version: top-level 'also_buy': [...], 'also_view': [...]
        # Ho tro ca hai de khong bi miss toan bo item-item triples.
        related = record.get("related", {})

        also_bought_list = (
            related.get("also_bought")
            or record.get("also_buy")
            or []
        )
        also_viewed_list = (
            related.get("also_viewed")
            or record.get("also_view")
            or []
        )

        for tgt_asin in also_bought_list:
            tgt_item = asin2new_item.get(str(tgt_asin))
            if tgt_item is not None and tgt_item != src_item:
                fwd_triples.add((src_item, REL_ALSO_BOUGHT, tgt_item))

        for tgt_asin in also_viewed_list:
            tgt_item = asin2new_item.get(str(tgt_asin))
            if tgt_item is not None and tgt_item != src_item:
                fwd_triples.add((src_item, REL_ALSO_VIEWED, tgt_item))

        # -- categories -------------------------------------------------------
        raw_cats = (
            record.get("categories")
            or record.get("category")
            or []
        )

        # Normalize: list of paths, moi path la list of str
        normalized_paths: List[List[str]] = []
        for cat_path in raw_cats:
            if isinstance(cat_path, list):
                path = [str(c).strip() for c in cat_path if c and str(c).strip()]
                if path:
                    normalized_paths.append(path)
            elif isinstance(cat_path, str):
                cat = cat_path.strip()
                if cat:
                    normalized_paths.append([cat])

        for path in normalized_paths:
            for cat_name in path:
                if cat_name not in category2eid:
                    category2eid[cat_name] = next_eid
                    next_eid += 1
                fwd_triples.add((src_item, REL_CATEGORY, category2eid[cat_name]))

        # -- brand ------------------------------------------------------------
        # [FIX-7] first_path: path dau tien (neu co) dung cho fallback leaf
        first_path = normalized_paths[0] if normalized_paths else []
        brand_val = _extract_brand(record, first_path)

        if brand_val:
            # [FIX-13] Assign brand entity ID ngay tai day trong vong lap chinh.
            # brand_pending chi defer viec ghi triple sau khi xong vong lap.
            if brand_val not in brand2eid:
                brand2eid[brand_val] = next_eid
                next_eid += 1
            brand_pending.append((src_item, brand_val))

    logger.info(
        f"       Pass 1 done: "
        f"{n_processed:,} records | {n_matched:,} matched"
    )

    # --- Brand triples -------------------------------------------------------
    for src_item, brand in brand_pending:
        fwd_triples.add((src_item, REL_BRAND, brand2eid[brand]))

    # --- Them inverse relations ----------------------------------------------
    # [FIX-5] rel_inv = rel_fwd + N_FORWARD_RELATIONS
    all_triples: List[Tuple[int, int, int]] = list(fwd_triples)
    inv_triples: List[Tuple[int, int, int]] = [
        (t, r + N_FORWARD_RELATIONS, h) for h, r, t in fwd_triples
    ]
    all_triples.extend(inv_triples)

    n_entities  = next_eid
    n_relations = N_FORWARD_RELATIONS * 2  # 8

    # --- Coverage ------------------------------------------------------------
    # [FIX-2] Coverage = ty le items thuc su co record trong gz file.
    # n_mapped = so ASIN co trong item_list.txt (= n_items sau remap -> 100%)
    # n_matched = so items thuc su tim thay record trong meta_Books.json.gz
    # Coverage dung nghia phai la n_matched / n_items.
    kg_coverage = n_matched / n_items if n_items > 0 else 0.0
    n_items_without_kg = n_items - n_matched

    # --- Log summary ---------------------------------------------------------
    logger.info("  [KG] Summary:")
    logger.info(f"       Item entities    : {n_items:,}")
    logger.info(f"       Category entities: {len(category2eid):,}")
    logger.info(f"       Brand entities   : {len(brand2eid):,}")
    logger.info(f"       Total entities   : {n_entities:,}")
    logger.info(f"       Forward relations: {N_FORWARD_RELATIONS}")
    logger.info(f"       Total relations  : {n_relations}  (fwd + inv)")
    logger.info(f"       Forward triples  : {len(fwd_triples):,}")
    logger.info(f"       Total triples    : {len(all_triples):,}  (fwd + inv)")
    logger.info(f"       KG coverage      : {kg_coverage:.4f}")
    logger.info(
        f"       Items without KG : {n_items_without_kg:,} "
        f"({n_items_without_kg / n_items * 100:.1f}%) -- "
        f"no record found in meta_Books.json.gz"
    )

    return all_triples, n_entities, n_relations, kg_coverage


def build_kg_variants(
    triples:     List[Tuple[int, int, int]],
    out_dir:     str,
    n_relations: int = 8,
) -> None:
    """
    Tach KG variants:
      kg_full.txt         -- tat ca 4 fwd + 4 inv relation
      kg_category.txt     -- rel 2 (has_category) + rel 6 (category_of)
      kg_brand.txt        -- rel 3 (has_brand) + rel 7 (brand_of)
      kg_item_item.txt    -- rel 0,1 (also_bought/viewed) + inv 4,5
    """
    REL_ALSO_BOUGHT = 0
    REL_ALSO_VIEWED = 1
    REL_CATEGORY    = 2
    REL_BRAND       = 3
    N = N_FORWARD_RELATIONS  # 4

    cat_triples       = [(h, r, t) for h, r, t in triples if r in (REL_CATEGORY, REL_CATEGORY + N)]
    brand_triples     = [(h, r, t) for h, r, t in triples if r in (REL_BRAND, REL_BRAND + N)]
    item_item_triples = [(h, r, t) for h, r, t in triples
                         if r in (REL_ALSO_BOUGHT, REL_ALSO_VIEWED,
                                   REL_ALSO_BOUGHT + N, REL_ALSO_VIEWED + N)]

    write_kg_file(os.path.join(out_dir, "kg_full.txt"),      triples)
    write_kg_file(os.path.join(out_dir, "kg_category.txt"),  cat_triples)
    write_kg_file(os.path.join(out_dir, "kg_brand.txt"),     brand_triples)
    write_kg_file(os.path.join(out_dir, "kg_item_item.txt"), item_item_triples)

    logger.info(
        f"  KG variants saved:\n"
        f"    kg_full.txt:      {len(triples):,}\n"
        f"    kg_category.txt:  {len(cat_triples):,}\n"
        f"    kg_brand.txt:     {len(brand_triples):,}\n"
        f"    kg_item_item.txt: {len(item_item_triples):,}"
    )


# ------------------------------------------------------------------------------
# Core CF pipeline steps
# ------------------------------------------------------------------------------

def five_core_filter(
    train_d:   Dict[int, List[int]],
    test_d:    Dict[int, List[int]],
    min_count: int = 5,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    5-core filtering tren union(train, test) cho den khi hoi tu.
    Giu user/item xuat hien >= min_count lan trong toan bo data.

    [FIX-4] Dedup all_pairs (dung set) truoc khi dem de tranh inflate count
    do trung lap trong file goc.
    """
    logger.info(f"  [Step 2] 5-core filtering (min={min_count}) ...")

    # Gop + dedup ngay bang set
    all_pairs: Set[Tuple[int, int]] = set()
    for u, items in train_d.items():
        for i in items:
            all_pairs.add((u, i))
    for u, items in test_d.items():
        for i in items:
            all_pairs.add((u, i))

    pairs_list = list(all_pairs)

    iteration = 0
    while True:
        iteration += 1
        user_cnt: Dict[int, int] = defaultdict(int)
        item_cnt: Dict[int, int] = defaultdict(int)
        for u, i in pairs_list:
            user_cnt[u] += 1
            item_cnt[i] += 1

        valid_u = {u for u, c in user_cnt.items() if c >= min_count}
        valid_i = {i for i, c in item_cnt.items() if c >= min_count}

        new_pairs = [(u, i) for u, i in pairs_list if u in valid_u and i in valid_i]
        if len(new_pairs) == len(pairs_list):
            break
        pairs_list = new_pairs

    valid_users = {u for u, _ in pairs_list}
    valid_items = {i for _, i in pairs_list}

    def _filter(d: Dict[int, List[int]]) -> Dict[int, List[int]]:
        r: Dict[int, List[int]] = {}
        for u, items in d.items():
            if u not in valid_users:
                continue
            fi = sorted({i for i in items if i in valid_items})  # dedup + sort
            if fi:
                r[u] = fi
        return r

    new_train = _filter(train_d)
    new_test  = _filter(test_d)

    n_tr = sum(len(v) for v in new_train.values())
    n_te = sum(len(v) for v in new_test.values())
    logger.info(
        f"    Converged (iter={iteration}): "
        f"{len(valid_users):,} users | {len(valid_items):,} items | "
        f"train={n_tr:,} | test={n_te:,}"
    )
    return new_train, new_test


def remap_ids(
    train_d: Dict[int, List[int]],
    test_d:  Dict[int, List[int]],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, int], Dict[int, int]]:
    """
    Remap user_id -> [0, N), item_id -> [0, M).
    Dung union(train, test) de dam bao nhat quan.
    """
    logger.info("  [Step 3] Remapping IDs -> [0, N) / [0, M)")

    all_users = sorted(set(train_d) | set(test_d))
    all_items = sorted(
        {i for v in train_d.values() for i in v}
        | {i for v in test_d.values()  for i in v}
    )

    user_map = {old: new for new, old in enumerate(all_users)}
    item_map = {old: new for new, old in enumerate(all_items)}

    def _remap(d: Dict[int, List[int]]) -> Dict[int, List[int]]:
        return {
            user_map[u]: sorted(item_map[i] for i in items)
            for u, items in d.items() if u in user_map
        }

    r_train = _remap(train_d)
    r_test  = _remap(test_d)

    logger.info(
        f"    {len(user_map):,} users [0..{len(user_map)-1}] | "
        f"{len(item_map):,} items [0..{len(item_map)-1}]"
    )
    return r_train, r_test, user_map, item_map


def split_val_from_train(
    train_d:   Dict[int, List[int]],
    val_ratio: float = 0.1,
    seed:      int   = 42,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Tach val tu train pool (10%, user-wise).

    No-timestamp protocol: shuffle(seed=seed), lay 10% CUOI lam val.
    [FIX-3] Lay arr[-n_val:] (cuoi) thay vi arr[:n_val] (dau) -- semantic
    dung hon: val la cac interaction "moi nhat" sau shuffle, tuong duong
    nhanh timestamp (lay 10% cuoi theo thoi gian).
    """
    logger.info(
        f"  [Step 4] Tach val tu train ({val_ratio:.0%} user-wise, "
        f"no-timestamp -> shuffle seed={seed}, lay 10% cuoi)"
    )
    rng = np.random.RandomState(seed)
    new_train: Dict[int, List[int]] = {}
    val:       Dict[int, List[int]] = {}
    skipped = 0

    for uid, items in train_d.items():
        arr = np.array(items, dtype=np.int64)
        rng.shuffle(arr)
        n = len(arr)
        if n < 2:
            new_train[uid] = arr.tolist()
            skipped += 1
            continue
        n_val = max(1, int(n * val_ratio))
        val[uid]       = arr[-n_val:].tolist()   # [FIX-3] 10% CUOI -> val
        new_train[uid] = arr[:-n_val].tolist()   # 90% dau -> train

    if skipped:
        logger.warning(f"    {skipped} users co <2 items -> val rong.")

    n_tr = sum(len(v) for v in new_train.values())
    n_va = sum(len(v) for v in val.values())
    logger.info(f"    Train: {n_tr:,} | Val: {n_va:,}")
    return new_train, val


def compute_stats(
    train_d: Dict[int, List[int]],
    val_d:   Dict[int, List[int]],
    test_d:  Dict[int, List[int]],
    n_users: int,
    n_items: int,
    val_ratio:    float           = 0.1,
    val_seed:     int             = 42,
    n_entities:   Optional[int]   = None,
    n_relations:  Optional[int]   = None,
    n_kg_triples: Optional[int]   = None,
    kg_coverage:  Optional[float] = None,
) -> dict:
    """
    Tinh thong ke dataset.

    [FIX-12] Bo sung density_train (chi tinh tren tap train) de de so sanh
    voi paper. density_all_splits tinh tren toan bo train+val+test.
    Bo sung n_items_without_kg de ro hon ve muc do phu KG.
    """
    n_tr = sum(len(v) for v in train_d.values())
    n_va = sum(len(v) for v in val_d.values())
    n_te = sum(len(v) for v in test_d.values())

    denom = n_users * n_items if n_users * n_items else 1
    density_train     = n_tr / denom
    density_all_splits = (n_tr + n_va + n_te) / denom

    stats: dict = {
        "n_users":           n_users,
        "n_items":           n_items,
        "n_train":           n_tr,
        "n_val":             n_va,
        "n_test":            n_te,
        "n_total":           n_tr + n_va + n_te,
        "density_train":     round(density_train,      8),
        "density_all_splits": round(density_all_splits, 8),
        "split_protocol":    "original_test_val_from_train_shuffle_last10pct",
        "val_ratio":         val_ratio,
        "val_seed":          val_seed,
    }
    if n_entities is not None:
        n_without_kg = n_items - round(kg_coverage * n_items) if kg_coverage is not None else None
        stats.update({
            "n_entities":        n_entities,
            "n_relations":       n_relations,
            "n_kg_triples":      n_kg_triples,
            "kg_coverage":       round(kg_coverage or 0.0, 4),
            "n_items_without_kg": n_without_kg,
            "kg_coverage_note":  (
                "fraction of items with at least one record in meta_Books.json.gz; "
                "items without KG coverage have no KG triples and rely solely on "
                "CF-side embeddings"
            ),
        })
    return stats


# ------------------------------------------------------------------------------
# Reproducibility check
# ------------------------------------------------------------------------------

def _fingerprint(d: Dict[int, List[int]]) -> str:
    """MD5 fingerprint cua mot interaction dict."""
    h = hashlib.md5()
    for u in sorted(d):
        h.update(f"{u}:{','.join(map(str, d[u]))}".encode())
    return h.hexdigest()


def verify_reproducibility(
    train_pool_d: Dict[int, List[int]],
    train_d:      Dict[int, List[int]],
    val_d:        Dict[int, List[int]],
    test_d:       Dict[int, List[int]],
    val_ratio:    float = 0.1,
    seed:         int   = 42,
    n_runs:       int   = 3,
) -> None:
    """
    Kiem tra reproducibility cua buoc split_val.

    [FIX-A / FIX-6] Nhan train_pool_d (train TRUOC khi split val) thay vi
    train_d (train SAU khi split val). Bug cu: goi split_val_from_train(train_d)
    -> lay 10% cua 90% con lai -> val size sai ~11%, fingerprint log khong
    khop voi data thuc te dung de train model.

    Cach verify dung:
      1. Chay lai split_val_from_train(train_pool_d, seed) n_runs lan
      2. So sanh fingerprint voi train_d va val_d da tinh o main pipeline
      3. Neu khop tat ca -> pipeline deterministic
    """
    logger.info(f"  [Step 7] Reproducibility check ({n_runs} runs) ...")

    expected_train_fp = _fingerprint(train_d)
    expected_val_fp   = _fingerprint(val_d)
    test_fp           = _fingerprint(test_d)

    for run in range(1, n_runs + 1):
        tr, va = split_val_from_train(train_pool_d, val_ratio=val_ratio, seed=seed)
        got_train_fp = _fingerprint(tr)
        got_val_fp   = _fingerprint(va)

        if got_train_fp != expected_train_fp or got_val_fp != expected_val_fp:
            raise RuntimeError(
                f"Reproducibility FAILED (run {run}):\n"
                f"  expected train fp: {expected_train_fp}\n"
                f"  got      train fp: {got_train_fp}\n"
                f"  expected val   fp: {expected_val_fp}\n"
                f"  got      val   fp: {got_val_fp}"
            )

    n_tr = sum(len(v) for v in train_d.values())
    n_va = sum(len(v) for v in val_d.values())
    n_te = sum(len(v) for v in test_d.values())
    logger.info(f"    train fp : {expected_train_fp}  (n={n_tr:,})")
    logger.info(f"    val   fp : {expected_val_fp}  (n={n_va:,})")
    logger.info(f"    test  fp : {test_fp}  (n={n_te:,})")
    logger.info("    Reproducibility PASSED")


# ------------------------------------------------------------------------------
# Dataset entry points
# ------------------------------------------------------------------------------

def preprocess_amazon_book(raw_dir: str, out_dir: str) -> None:
    """
    Preprocessing Amazon-Book (LightGCN repo + meta_Books.json.gz).
    """
    logger.info("=" * 65)
    logger.info("PREPROCESSING: Amazon-Book")
    logger.info("  CF source : LightGCN-PyTorch repo")
    logger.info("  KG source : meta_Books")
    logger.info("  Protocol  : original test split, val carved from train")
    logger.info("=" * 65)

    ds_dir = os.path.join(raw_dir, "amazon-book")
    required = {
        "train.txt":          os.path.join(ds_dir, "train.txt"),
        "test.txt":           os.path.join(ds_dir, "test.txt"),
        "item_list.txt":      os.path.join(ds_dir, "item_list.txt"),
        "user_list.txt":      os.path.join(ds_dir, "user_list.txt"),
        "meta_Books.json.gz": os.path.join(ds_dir, "meta_Books.json.gz"),
    }
    missing = [k for k, v in required.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(
            f"Thieu file(s): {missing}\n"
            "Chay: python scripts/download_data.py --dataset amazon-book"
        )

    os.makedirs(out_dir, exist_ok=True)

    # -- Step 1: doc ----------------------------------------------------------
    logger.info("  [Step 1] Doc raw interactions ...")
    raw_train = pairs_to_dict(read_lightgcn_interaction(required["train.txt"]))
    raw_test  = pairs_to_dict(read_lightgcn_interaction(required["test.txt"]))
    logger.info(
        f"    Raw train: {sum(len(v) for v in raw_train.values()):,} pairs | "
        f"{len(raw_train):,} users"
    )
    logger.info(
        f"    Raw test:  {sum(len(v) for v in raw_test.values()):,} pairs | "
        f"{len(raw_test):,} users"
    )

    # Doc item_list -- {remap_id -> ASIN}  [FIX-1]
    logger.info("    Doc item_list.txt ...")
    item2asin = read_item_list_lightgcn(required["item_list.txt"])
    logger.info(f"    item_list: {len(item2asin):,} entries")

    # -- Steps 2-3: filter + remap --------------------------------------------
    raw_train, raw_test = five_core_filter(raw_train, raw_test)
    train_d, test_d, user_map, item_map = remap_ids(raw_train, raw_test)
    n_users = len(user_map)
    n_items = len(item_map)

    # -- Step 4: tach val -----------------------------------------------------
    # Luu lai train_pool_d TRUOC khi split -- dung cho reproducibility check
    train_pool_d = {u: list(items) for u, items in train_d.items()}
    train_d, val_d = split_val_from_train(train_d, val_ratio=0.1, seed=42)

    # -- Step 7: reproducibility ----------------------------------------------
    # [FIX-A] Truyen train_pool_d (TRUOC split) + train_d/val_d (SAU split)
    verify_reproducibility(train_pool_d, train_d, val_d, test_d, val_ratio=0.1, seed=42)

    # -- Step 5: xay KG -------------------------------------------------------
    triples, n_entities, n_relations, kg_coverage = build_kg_from_meta(
        gz_path   = required["meta_Books.json.gz"],
        item2asin = item2asin,   # {remap_id -> ASIN}  [FIX-1]
        item_map  = item_map,    # {remap_id -> new_item_id}
        n_items   = n_items,
    )

    # -- Step 6: stats --------------------------------------------------------
    stats = compute_stats(
        train_d, val_d, test_d, n_users, n_items,
        val_ratio    = 0.1,
        val_seed     = 42,
        n_entities   = n_entities,
        n_relations  = n_relations,
        n_kg_triples = len(triples),
        kg_coverage  = kg_coverage,
    )
    logger.info("  [Step 6] Dataset statistics:")
    for k, v in stats.items():
        logger.info(f"    {k:<35} = {v}")

    # -- Step 8: luu ----------------------------------------------------------
    logger.info("  [Step 8] Saving ...")

    # [FIX-10] mkdir mot lan o day, khong goi trong write_interaction_file
    os.makedirs(out_dir, exist_ok=True)

    write_interaction_file(os.path.join(out_dir, "train.txt"), train_d)
    write_interaction_file(os.path.join(out_dir, "val.txt"),   val_d)
    write_interaction_file(os.path.join(out_dir, "test.txt"),  test_d)

    # item2entity: new_item_id -> entity_id (== new_item_id theo thiet ke)
    item2entity = {i: i for i in range(n_items)}
    with open(os.path.join(out_dir, "item2entity.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in item2entity.items()}, f)

    # id_maps de truy vet
    with open(os.path.join(out_dir, "id_maps.json"), "w", encoding="utf-8") as f:
        json.dump({
            "user_map": {str(k): v for k, v in user_map.items()},
            "item_map": {str(k): v for k, v in item_map.items()},
        }, f)

    # KG metadata
    kg_meta = {
        "n_entities":  n_entities,
        "n_relations": n_relations,
        "n_triples":   len(triples),
        "relations": {
            "0": "also_bought    (item->item,     forward)",
            "1": "also_viewed    (item->item,     forward)",
            "2": "has_category   (item->category, forward)",
            "3": "has_brand      (item->brand,    forward)",
            "4": "also_bought_by (item<-item,     inverse of 0)",
            "5": "also_viewed_by (item<-item,     inverse of 1)",
            "6": "category_of    (category<-item, inverse of 2)",
            "7": "brand_of       (brand<-item,    inverse of 3)",
        },
        "entity_ranges": {
            "items":      f"[0, {n_items})",
            "categories": f"[{n_items}, n_items+n_categories)",
            "brands":     f"[n_items+n_categories, {n_entities})",
        },
        "note": "n_relations=8 (4 fwd + 4 inv) -- compatible with KGAT/KGCL/RippleNet",
    }
    with open(os.path.join(out_dir, "kg_meta.json"), "w", encoding="utf-8") as f:
        json.dump(kg_meta, f, indent=2)

    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    build_kg_variants(triples, out_dir)

    logger.info(f"  Amazon-Book processed -> {out_dir}")
    logger.info("=" * 65)


def preprocess_yelp2018(raw_dir: str, out_dir: str) -> None:
    """Preprocessing Yelp2018 (CF only, khong KG)."""
    logger.info("=" * 65)
    logger.info("PREPROCESSING: Yelp2018  (SimGCL / QRec repo, CF only)")
    logger.info("=" * 65)

    ds_dir  = os.path.join(raw_dir, "yelp2018")
    train_p = os.path.join(ds_dir, "train.txt")
    test_p  = os.path.join(ds_dir, "test.txt")
    for p in [train_p, test_p]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Thieu: {p}\n"
                "Chay: python scripts/download_data.py --dataset yelp2018"
            )

    os.makedirs(out_dir, exist_ok=True)

    logger.info("  [Step 1] Doc raw interactions ...")
    raw_train = pairs_to_dict(read_lightgcn_interaction(train_p))
    raw_test  = pairs_to_dict(read_lightgcn_interaction(test_p))

    raw_train, raw_test = five_core_filter(raw_train, raw_test)
    train_d, test_d, user_map, item_map = remap_ids(raw_train, raw_test)
    n_users = len(user_map)
    n_items = len(item_map)

    train_pool_d = {u: list(items) for u, items in train_d.items()}
    train_d, val_d = split_val_from_train(train_d, val_ratio=0.1, seed=42)

    # [FIX-A] Truyen train_pool_d dung
    verify_reproducibility(train_pool_d, train_d, val_d, test_d, val_ratio=0.1, seed=42)

    stats = compute_stats(
        train_d, val_d, test_d, n_users, n_items,
        val_ratio = 0.1,
        val_seed  = 42,
    )
    logger.info("  [Step 6] Dataset statistics:")
    for k, v in stats.items():
        logger.info(f"    {k:<35} = {v}")

    write_interaction_file(os.path.join(out_dir, "train.txt"), train_d)
    write_interaction_file(os.path.join(out_dir, "val.txt"),   val_d)
    write_interaction_file(os.path.join(out_dir, "test.txt"),  test_d)

    with open(os.path.join(out_dir, "id_maps.json"), "w", encoding="utf-8") as f:
        json.dump({
            "user_map": {str(k): v for k, v in user_map.items()},
            "item_map": {str(k): v for k, v in item_map.items()},
        }, f)

    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"  Yelp2018 processed -> {out_dir}")
    logger.info("=" * 65)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocessing -- Amazon-Book (LightGCN repo + meta_Books.json.gz KG) "
            "& Yelp2018"
        )
    )
    parser.add_argument(
        "--dataset", choices=["amazon-book", "yelp2018", "all"], default="all"
    )
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)

    if args.dataset in ("amazon-book", "all"):
        preprocess_amazon_book(
            raw_dir=args.raw_dir,
            out_dir=os.path.join(args.out_dir, "amazon-book"),
        )

    if args.dataset in ("yelp2018", "all"):
        preprocess_yelp2018(
            raw_dir=args.raw_dir,
            out_dir=os.path.join(args.out_dir, "yelp2018"),
        )

    logger.info("Preprocessing hoan thanh.")


if __name__ == "__main__":
    main()
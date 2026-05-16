"""
scripts/download_data.py
─────────────────────────────────────────────────────────────────────────────
Tải dữ liệu thô cho Amazon-Book (LightGCN repo) và Yelp2018 (SimGCL repo).

Amazon-Book — nguồn: LightGCN-PyTorch repo
  https://github.com/gusye1234/LightGCN-PyTorch
  Files CF   : train.txt, test.txt, user_list.txt, item_list.txt
  File meta  : meta_Books.json.gz  (Amazon product metadata — dùng để xây KG)
               Ưu tiên dùng Amazon 2018 (có trường brand/publisher đầy đủ):
                 https://nijianmo.github.io/amazon/index.html
                 → Books → metadata → điền form → tải meta_Books.json.gz
               Fallback: McAuley 2014 categoryFiles (không có brand, sẽ dùng
                 leaf_category làm proxy brand):
                 https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/

Yelp2018 — nguồn: SimGCL / QRec repo
  https://github.com/Coder-Yu/QRec
  Files: train.txt, test.txt

QUAN TRỌNG về meta_Books.json.gz:
  - Amazon 2018 (khuyến nghị): yêu cầu điền Google Form để lấy link tải.
    Sau khi có file, đổi tên thành meta_Books.json.gz và đặt vào
    data/raw/amazon-book/
  - Amazon 2014 (fallback): có thể tải trực tiếp từ McAuley lab UCSD.
    Không có trường brand → script sẽ dùng leaf_category làm proxy.

Usage:
  python scripts/download_data.py --dataset amazon-book
  python scripts/download_data.py --dataset yelp2018
  python scripts/download_data.py --dataset all
  python scripts/download_data.py --dataset amazon-book --check_only
"""

import argparse
import gzip
import os
import sys
import urllib.request
import urllib.error
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_script_logger

logger = get_script_logger("download_data")

# ─────────────────────────────────────────────────────────────────────────────
# URLs
# ─────────────────────────────────────────────────────────────────────────────

_LGCN_RAW = (
    "https://raw.githubusercontent.com/"
    "gusye1234/LightGCN-PyTorch/master/data/amazon-book"
)

_QREC_RAW = (
    "https://raw.githubusercontent.com/"
    "Coder-Yu/QRec/master/dataset/yelp2018"
)

# meta_Books.json.gz — thử nhiều mirror theo thứ tự
# Amazon 2018 (có trường brand/publisher đầy đủ) — link nhận được sau khi
# điền Google Form tại https://nijianmo.github.io/amazon/
# Fallback về Amazon 2014 (không có brand → preprocess.py dùng leaf_category).
_META_BOOKS_URLS: List[str] = [
    # ── Amazon 2018 (primary) ──────────────────────────
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Books.json.gz",
    # ── Amazon 2014 (fallback) ─────────────────────────────
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/meta_Books.json.gz",
]

# Thông báo khi tải thành công từ 2018 vs 2014
_META_BOOKS_2018_INSTRUCTIONS = """
  ╔══════════════════════════════════════════════════════════════╗
  ║  Script sẽ tự tải Amazon 2018.         ║
  ║  Nếu tất cả mirror đều thất bại, tải thủ công:             ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  1. Truy cập: https://nijianmo.github.io/amazon/            ║
  ║  2. Mục "Books" → cột "metadata" → click link              ║
  ║  3. Điền Google Form → nhận link tải                        ║
  ║  4. Tải meta_Books.json.gz, đặt vào:                        ║
  ║     data/raw/amazon-book/meta_Books.json.gz                  ║
  ╚══════════════════════════════════════════════════════════════╝
"""

DATASETS: Dict[str, dict] = {
    "amazon-book": {
        "cf_files": {
            # fname: url
            "train.txt":     f"{_LGCN_RAW}/train.txt",
            "test.txt":      f"{_LGCN_RAW}/test.txt",
            "user_list.txt": f"{_LGCN_RAW}/user_list.txt",
            "item_list.txt": f"{_LGCN_RAW}/item_list.txt",
        },
        # meta_Books.json.gz tải riêng (lớn, nhiều mirror)
        "meta_file": "meta_Books.json.gz",
        "required": [
            "train.txt", "test.txt",
            "item_list.txt", "user_list.txt",
            "meta_Books.json.gz",
        ],
        "manual_instructions": """
Amazon-Book — tải thủ công:

  [CF files] Từ LightGCN-PyTorch repo:
    https://github.com/gusye1234/LightGCN-PyTorch/tree/master/data/amazon-book
    Tải: train.txt, test.txt, item_list.txt, user_list.txt
    Đặt vào: data/raw/amazon-book/

  [Meta file — Amazon 2018, CÓ brand/publisher] ← Khuyến nghị
    Direct URL: https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Books.json.gz
    Hoặc điền form tại: https://nijianmo.github.io/amazon/ → Books → metadata
    Đặt vào: data/raw/amazon-book/meta_Books.json.gz

  [Meta file — Amazon 2014, fallback nếu 2018 không tải được]
    https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/meta_Books.json.gz
    Không có trường brand → preprocess.py tự dùng leaf_category làm proxy.
""",
    },
    "yelp2018": {
        "cf_files": {
            "train.txt": f"{_QREC_RAW}/train.txt",
            "test.txt":  f"{_QREC_RAW}/test.txt",
        },
        "meta_file": None,
        "required": ["train.txt", "test.txt"],
        "manual_instructions": """
Yelp2018 — tải thủ công:
  https://github.com/Coder-Yu/QRec/tree/master/dataset/yelp2018
  Tải: train.txt, test.txt → đặt vào data/raw/yelp2018/
""",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Core download helpers
# ─────────────────────────────────────────────────────────────────────────────

def _download_file(url: str, dest_path: str, chunk_mb: int = 8) -> bool:
    """
    Tải file từ URL về dest_path với progress log mỗi 100 MB.
    Trả về True nếu thành công.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, \
             open(dest_path, "wb") as f:

            total = int(resp.headers.get("Content-Length", 0))
            total_mb = total / 1024 / 1024
            downloaded = 0
            chunk = chunk_mb * 1024 * 1024
            last_log_mb = 0

            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                dl_mb = downloaded / 1024 / 1024
                if dl_mb - last_log_mb >= 100 or (total and downloaded >= total):
                    pct = f"{dl_mb/total_mb:.0%}" if total_mb > 0 else ""
                    logger.info(
                        f"    ... {dl_mb:,.0f} MB"
                        + (f" / {total_mb:,.0f} MB  ({pct})" if total_mb > 0 else "")
                    )
                    last_log_mb = dl_mb

        size_mb = os.path.getsize(dest_path) / 1024 / 1024
        logger.info(f"    ✓  {os.path.basename(dest_path)}  ({size_mb:,.1f} MB)")
        return True

    except urllib.error.HTTPError as e:
        logger.warning(f"    HTTP {e.code}: {url}")
    except urllib.error.URLError as e:
        logger.warning(f"    URL error: {e.reason}  [{url}]")
    except Exception as e:
        logger.warning(f"    Download failed: {e}")

    # Xoá file hỏng
    if os.path.exists(dest_path) and os.path.getsize(dest_path) == 0:
        os.remove(dest_path)
    return False


def _verify_gz(path: str) -> bool:
    """Kiểm tra file .gz có đọc được không (không giải nén toàn bộ)."""
    try:
        with gzip.open(path, "rb") as f:
            f.read(1024)
        return True
    except Exception:
        return False


def _download_with_mirrors(urls: List[str], dest_path: str) -> bool:
    """Thử tải từ nhiều URL mirror, trả về True khi thành công."""
    for i, url in enumerate(urls, 1):
        logger.info(f"    Mirror {i}/{len(urls)}: {url}")
        ok = _download_file(url, dest_path)
        if ok:
            # Kiểm tra tính hợp lệ nếu là .gz
            if dest_path.endswith(".gz"):
                if _verify_gz(dest_path):
                    return True
                else:
                    logger.warning("    File .gz bị lỗi, thử mirror tiếp theo ...")
                    os.remove(dest_path)
            else:
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level check & download
# ─────────────────────────────────────────────────────────────────────────────

def check_dataset(name: str, raw_dir: str) -> bool:
    """Kiểm tra tất cả required files có mặt và không rỗng."""
    ds  = DATASETS[name]
    ds_dir = os.path.join(raw_dir, name)
    all_ok = True

    for fname in ds["required"]:
        path = os.path.join(ds_dir, fname)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            size_mb = os.path.getsize(path) / 1024 / 1024
            logger.info(f"  ✓  {fname:<35}  ({size_mb:,.1f} MB)")
        else:
            logger.warning(f"  ✗  {fname:<35}  MISSING / EMPTY")
            all_ok = False

    return all_ok


def attempt_download(name: str, raw_dir: str) -> None:
    """Tải tự động tất cả files của dataset."""
    ds     = DATASETS[name]
    ds_dir = os.path.join(raw_dir, name)
    os.makedirs(ds_dir, exist_ok=True)
    failed: List[str] = []

    # ── 1. CF files (nhỏ, từ GitHub) ─────────────────────────────────────────
    for fname, url in ds.get("cf_files", {}).items():
        dest = os.path.join(ds_dir, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            logger.info(f"  Đã có: {fname}")
            continue
        logger.info(f"  Đang tải: {fname}")
        if not _download_file(url, dest):
            failed.append(fname)

    # ── 2. meta_Books.json.gz (lớn, nhiều mirror) ────────────────────────────
    meta_fname = ds.get("meta_file")
    if meta_fname:
        dest = os.path.join(ds_dir, meta_fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 0 and _verify_gz(dest):
            logger.info(f"  Đã có: {meta_fname}")
        else:
            logger.info(
                f"  Đang tải: {meta_fname}  (~2 GB, thử META_BOOKS (2018) trước)\n"
                f"  {_META_BOOKS_2018_INSTRUCTIONS}"
            )
            if name == "amazon-book":
                ok = _download_with_mirrors(_META_BOOKS_URLS, dest)
                if not ok:
                    failed.append(meta_fname)
                    logger.warning(
                        "  ✗  meta_Books.json.gz không tải được tự động.\n"
                        "     Vui lòng tải thủ công theo hướng dẫn bên dưới."
                    )

    if failed:
        logger.warning(
            f"\n  Không tải được: {failed}\n"
            f"{ds['manual_instructions']}"
        )
    else:
        logger.info(f"  ✓ Tất cả files của {name} đã sẵn sàng.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tải dữ liệu thô — Amazon-Book (LightGCN repo) + Yelp2018"
    )
    parser.add_argument(
        "--dataset",
        choices=["amazon-book", "yelp2018", "all"],
        default="all",
    )
    parser.add_argument("--raw_dir", default="/data/phuongtran/raw") # data/raw
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Chỉ kiểm tra file, không tải.",
    )
    return parser.parse_args()


def main() -> None:
    args  = parse_args()
    names = ["amazon-book", "yelp2018"] if args.dataset == "all" else [args.dataset]

    for name in names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {name}")
        logger.info(f"{'='*60}")

        present = check_dataset(name, args.raw_dir)

        if present:
            logger.info(f"✓ {name} đầy đủ — sẵn sàng preprocess.")
        elif args.check_only:
            logger.warning(f"✗ {name} thiếu file(s). Bỏ --check_only để tải.")
        else:
            logger.info("Đang thử tải tự động ...")
            attempt_download(name, args.raw_dir)
            present = check_dataset(name, args.raw_dir)
            if not present:
                logger.warning(
                    f"✗ {name} chưa đầy đủ.\n"
                    + DATASETS[name]["manual_instructions"]
                )


if __name__ == "__main__":
    main()
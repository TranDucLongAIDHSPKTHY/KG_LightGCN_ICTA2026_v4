"""
utils/logger.py
─────────────────────────────────────────────────────────────────────────────
Hệ thống logging có cấu trúc cho toàn bộ project.

Thiết kế:
  ┌─ results/logs/
  │   ├─ preprocess/
  │   │   └─ preprocess_20260507_102030.log
  │   ├─ amazon-book/
  │   │   ├─ lightgcn/
  │   │   │   ├─ seed42/
  │   │   │   │   ├─ train.log          ← toàn bộ quá trình training
  │   │   │   │   └─ epoch_metrics.tsv  ← epoch | loss | recall@20 | ...
  │   │   │   └─ seed0/
  │   │   ├─ kg_lightgcn_cl/
  │   │   │   └─ seed42/
  │   │   │       ├─ train.log
  │   │   │       └─ epoch_metrics.tsv
  │   │   └─ ...
  │   └─ yelp2018/
  │       └─ ...

Module-level logger (không có file handler) dùng cho import warnings,
errors, setup messages — không lẫn vào log training.

Run-level logger (có file handler riêng) gắn với 1 cặp (model, dataset, seed).
"""

import csv
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Formatter chuẩn
# ─────────────────────────────────────────────────────────────────────────────

_CONSOLE_FMT = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_FILE_FMT = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ─────────────────────────────────────────────────────────────────────────────
# get_logger — module-level logger (console only, no file)
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Trả về module-level logger với console handler.
    Dùng cho các module như preprocess, evaluator, kg_dataset…
    KHÔNG ghi file — tránh lẫn log setup với log training.

    Args:
        name:  Tên logger (thường là module name, vd 'preprocess').
        level: Logging level.

    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(_CONSOLE_FMT)
    logger.addHandler(ch)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# get_run_logger — per-run logger (console + file riêng theo model/dataset/seed)
# ─────────────────────────────────────────────────────────────────────────────

def get_run_logger(
    model_name: str,
    dataset_name: str,
    seed: int,
    base_log_dir: str = "results/logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Tạo logger riêng cho một run (model + dataset + seed).
    Ghi log vào:
        <base_log_dir>/<dataset_name>/<model_name>/seed<seed>/train.log

    Logger name = "<model_name>_<dataset_name>_seed<seed>" → unique, không đụng nhau.

    Args:
        model_name:   Tên model, vd 'lightgcn', 'kg_lightgcn_cl'.
        dataset_name: Tên dataset, vd 'amazon-book', 'yelp2018'.
        seed:         Seed của run này.
        base_log_dir: Thư mục gốc logs.
        level:        Logging level.

    Returns:
        Logger instance có file handler ghi vào thư mục riêng.
    """
    logger_name = f"{model_name}_{dataset_name}_seed{seed}"
    logger = logging.getLogger(logger_name)

    # Đã được tạo rồi thì trả về luôn
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(_CONSOLE_FMT)
    logger.addHandler(ch)

    # File handler — thư mục: <base_log_dir>/<dataset>/<model>/seed<N>/
    run_log_dir = os.path.join(base_log_dir, dataset_name, model_name, f"seed{seed}")
    os.makedirs(run_log_dir, exist_ok=True)
    log_path = os.path.join(run_log_dir, "train.log")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(_FILE_FMT)
    logger.addHandler(fh)

    return logger


def get_script_logger(
    script_name: str,
    base_log_dir: str = "results/logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Logger dành cho scripts (preprocess, build_cold_split…).
    Ghi vào: <base_log_dir>/<script_name>/run_<timestamp>.log

    Args:
        script_name:  Tên script, vd 'preprocess', 'build_cold_split'.
        base_log_dir: Thư mục gốc logs.
        level:        Logging level.
    """
    logger_name = f"script_{script_name}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(_CONSOLE_FMT)
    logger.addHandler(ch)

    script_log_dir = os.path.join(base_log_dir, script_name)
    os.makedirs(script_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(script_log_dir, f"run_{timestamp}.log")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(_FILE_FMT)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# EpochLogger — ghi per-epoch metrics vào TSV riêng + logger
# ─────────────────────────────────────────────────────────────────────────────

class EpochLogger:
    """
    Ghi per-epoch training metrics theo hai kênh:
      1. Logger (INFO) → xuất hiện trong train.log và console
      2. TSV file riêng → dễ import vào pandas/Excel để vẽ đồ thị

    Cấu trúc TSV:
        epoch  loss  recall@20  ndcg@20  hr@10  ndcg@10  time_s
        1      0.69  0.0123     0.0089   0.0234 0.0078   12.3
        ...

    File lưu tại: <run_log_dir>/epoch_metrics.tsv
    """

    def __init__(
        self,
        run_logger: logging.Logger,
        model_name: str,
        dataset_name: str,
        seed: int,
        base_log_dir: str = "results/logs",
    ) -> None:
        """
        Args:
            run_logger:   Logger instance từ get_run_logger().
            model_name:   Tên model.
            dataset_name: Tên dataset.
            seed:         Seed của run này.
            base_log_dir: Thư mục gốc logs.
        """
        self.logger = run_logger
        self._header_written = False

        # TSV file path
        run_log_dir = os.path.join(base_log_dir, dataset_name, model_name, f"seed{seed}")
        os.makedirs(run_log_dir, exist_ok=True)
        self._tsv_path = os.path.join(run_log_dir, "epoch_metrics.tsv")
        self._tsv_file = open(self._tsv_path, "w", newline="", encoding="utf-8")
        self._tsv_writer: Optional[csv.DictWriter] = None

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.seed = seed

    def log(
        self,
        epoch: int,
        loss: float,
        metrics: Dict[str, float],
        time_s: float = 0.0,
    ) -> None:
        """
        Ghi một dòng epoch vào cả logger và TSV.

        Args:
            epoch:   Epoch hiện tại.
            loss:    Training loss.
            metrics: Dict metric_name → value (val metrics).
            time_s:  Thời gian epoch (giây).
        """
        # Khởi tạo TSV writer lần đầu (sau khi biết tên các metrics)
        if not self._header_written:
            fieldnames = ["epoch", "loss"] + sorted(metrics.keys()) + ["time_s"]
            self._tsv_writer = csv.DictWriter(
                self._tsv_file, fieldnames=fieldnames, delimiter="\t"
            )
            self._tsv_writer.writeheader()
            self._tsv_file.flush()
            self._header_written = True

        # Ghi TSV
        row = {"epoch": epoch, "loss": f"{loss:.6f}", "time_s": f"{time_s:.1f}"}
        row.update({k: f"{v:.6f}" for k, v in metrics.items()})
        self._tsv_writer.writerow(row)
        self._tsv_file.flush()

        # Ghi logger (dạng dễ đọc)
        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        self.logger.info(
            f"[Epoch {epoch:>4}]  loss={loss:.4f}  {metrics_str}  ({time_s:.1f}s)"
        )

    def close(self) -> None:
        """Đóng TSV file."""
        if self._tsv_file and not self._tsv_file.closed:
            self._tsv_file.close()

    def __del__(self) -> None:
        self.close()

    @property
    def tsv_path(self) -> str:
        return self._tsv_path


# ─────────────────────────────────────────────────────────────────────────────
# RunSummaryLogger — ghi tóm tắt kết quả cuối run
# ─────────────────────────────────────────────────────────────────────────────

class RunSummaryLogger:
    """
    Ghi summary JSON sau mỗi seed run.
    File: <run_log_dir>/summary.json

    Nội dung:
    {
      "model": "kg_lightgcn_cl",
      "dataset": "amazon-book",
      "seed": 42,
      "best_epoch": 87,
      "val_recall@20": 0.1234,
      "test_metrics": { "recall@20": 0.1245, ... },
      "total_time_s": 1234.5
    }
    """

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        seed: int,
        base_log_dir: str = "results/logs",
    ) -> None:
        run_log_dir = os.path.join(base_log_dir, dataset_name, model_name, f"seed{seed}")
        os.makedirs(run_log_dir, exist_ok=True)
        self._path = os.path.join(run_log_dir, "summary.json")
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.seed = seed

    def save(
        self,
        best_epoch: int,
        val_metric: float,
        test_metrics: Dict[str, float],
        total_time_s: float,
        extra: Optional[Dict] = None,
    ) -> None:
        """Ghi summary JSON cho run này."""
        import json

        data = {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "seed": self.seed,
            "best_epoch": best_epoch,
            "val_best_metric": round(val_metric, 6),
            "test_metrics": {k: round(v, 6) for k, v in test_metrics.items()},
            "total_time_s": round(total_time_s, 1),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            data.update(extra)

        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @property
    def path(self) -> str:
        return self._path

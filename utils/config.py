"""
utils/config.py
─────────────────────────────────────────────────────────────────────────────
Config loader cho KG-LightGCN.

Thứ tự ưu tiên (cao → thấp):
  1. CLI --override flags
  2. Model-specific YAML  (configs/model/<model>.yaml)
  3. Base YAML            (configs/base.yaml)
  4. Environment variable  (.env hoặc shell export)
  5. Hardcoded default

Path resolution:
  DATA_ROOT   → dataset directory (thay data/processed)
  RESULTS_ROOT → logs + checkpoints + tables directory (thay results/)
  NUM_WORKERS  → DataLoader workers
  DEVICE       → cpu | cuda | auto
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ── .env loader (không cần python-dotenv) ────────────────────────────────────

def _load_dotenv(path: str = ".env") -> None:
    """
    Đọc file .env và set vào os.environ nếu key chưa tồn tại.
    Hỗ trợ: KEY=value, KEY="value", # comment, dòng trống.
    Không ghi đè biến môi trường đã set từ shell.
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    with env_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:   # shell env thắng .env
                os.environ[key] = val


def _resolve_paths(cfg: dict) -> dict:
    """
    Thay thế data_dir, log_dir, checkpoint_dir, result_dir
    bằng giá trị từ env vars nếu có.

    Env vars được đọc (sau khi load .env):
      DATA_ROOT     → gốc cho dataset (thay dataset.data_dir)
      RESULTS_ROOT  → gốc cho results (thay logging.*)
      NUM_WORKERS   → train.num_workers
      DEVICE        → train.device
    """
    data_root = os.getenv("DATA_ROOT")
    results_root = os.getenv("RESULTS_ROOT")
    num_workers = os.getenv("NUM_WORKERS")
    device = os.getenv("DEVICE")

    if data_root:
        cfg.setdefault("dataset", {})["data_dir"] = data_root

    if results_root:
        cfg.setdefault("logging", {}).update({
            "log_dir":        os.path.join(results_root, "logs"),
            "checkpoint_dir": os.path.join(results_root, "checkpoints"),
            "result_dir":     os.path.join(results_root, "tables"),
        })

    if num_workers is not None:
        cfg.setdefault("train", {})["num_workers"] = int(num_workers)

    if device:
        cfg.setdefault("train", {})["device"] = device

    return cfg


# ── Deep merge ────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def load_config(
    base_path: str = "configs/base.yaml",
    model_config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    dotenv_path: str = ".env",
) -> dict:
    """
    Load config theo thứ tự ưu tiên:
      base YAML → model YAML → .env / env vars → CLI overrides.

    Args:
        base_path:         Path tới configs/base.yaml.
        model_config_path: Path tới configs/model/<model>.yaml (optional).
        overrides:         Dict từ CLI --override flags.
        dotenv_path:       Path tới .env (mặc định: ".env" ở root project).

    Returns:
        Merged config dict.
    """
    # 1. Load .env trước để env vars sẵn sàng cho path resolution
    _load_dotenv(dotenv_path)

    # 2. Base config
    with open(base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # 3. Model-specific config
    if model_config_path and os.path.exists(model_config_path):
        with open(model_config_path, "r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, model_cfg)

    # 4. Resolve paths từ env vars (DATA_ROOT, RESULTS_ROOT, ...)
    cfg = _resolve_paths(cfg)

    # 5. CLI overrides (cao nhất)
    if overrides:
        for key, value in overrides.items():
            keys = key.split(".")
            d = cfg
            for k in keys[:-1]:
                if k not in d or not isinstance(d[k], dict):
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value

    return cfg


def save_config(cfg: dict, path: str) -> None:
    """Lưu config ra YAML (dùng để reproduce experiment)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=True, allow_unicode=True)


class Config:
    """Wrapper cho phép truy cập config theo dot notation: cfg.train.lr."""

    def __init__(self, data: dict) -> None:
        for k, v in data.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        result = {}
        for k, v in self.__dict__.items():
            result[k] = v.to_dict() if isinstance(v, Config) else v
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"

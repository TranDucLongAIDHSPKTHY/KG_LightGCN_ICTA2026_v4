# ─────────────────────────────────────────────────────────────────────────────
# KG-LightGCN ICTA2026 — Dockerfile
#
# Build GPU  (RTX 3050, CUDA 11.8):
#   docker build -t kg-lightgcn:gpu .
#
# Build CPU-only (mọi máy, không cần GPU):
#   docker build --build-arg MODE=cpu -t kg-lightgcn:cpu .
#
# Dataset và results KHÔNG copy vào image.
# Chúng được bind mount khi chạy container (xem docker-compose.yml).
#   Dataset  → lưu trên HDD, mount vào /datasets
#   Results  → lưu trên SSD, mount vào /results
# ─────────────────────────────────────────────────────────────────────────────

ARG MODE=gpu

# ── GPU base (PyTorch 2.1.2 + CUDA 11.8) ─────────────────────────────────────
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime AS base-gpu

# ── CPU-only base ─────────────────────────────────────────────────────────────
FROM python:3.10-slim AS base-cpu
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    --index-url https://download.pytorch.org/whl/cpu

# ── Final stage ───────────────────────────────────────────────────────────────
FROM base-${MODE} AS final

LABEL maintainer="TranDucLong <AIDHSPKTHY>" \
      description="KG-LightGCN ICTA2026" \
      version="1.2"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── Cài Python deps trước (tận dụng layer cache) ─────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code (không copy data/, results/ — xem .dockerignore) ────────
COPY . .

# ── Tạo mount points (thư mục trống, nội dung đến từ bind mount) ─────────────
# /datasets → bind mount từ HDD (dataset nằm ở đây)
# /results  → bind mount từ SSD (logs, checkpoints, tables)
RUN mkdir -p /datasets /results/logs /results/checkpoints /results/tables

# ── Environment defaults (overridable qua .env hoặc -e flag) ─────────────────
# Dataset và results trỏ vào mount points
ENV DATA_ROOT=/datasets
ENV RESULTS_ROOT=/results
ENV DEVICE=auto
ENV NUM_WORKERS=4
# CUDA memory tối ưu cho RTX 3050 (4GB VRAM)
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=60s --timeout=30s --start-period=15s \
    CMD python3 -c "\
import torch; \
from models.lightgcn import LightGCN; \
from models.kgat import KGAT; \
from models.kg_lightgcn import KGLightGCN, KGLightGCNCL; \
print('GPU' if torch.cuda.is_available() else 'CPU', '— all models OK')" \
    || exit 1

CMD ["python3", "main.py", "--help"]

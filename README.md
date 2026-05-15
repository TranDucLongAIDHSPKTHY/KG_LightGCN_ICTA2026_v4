# KG-LightGCN ICTA2026

**Câu hỏi trung tâm:** Trong Collaborative Filtering, khi dùng cùng backbone LightGCN và InfoNCE loss, dạng structured side information nào tạo contrastive signal hiệu quả hơn: flat category, taxonomy, hay knowledge graph?

---

## Mô hình

| Model                             | Loại               | Venue      |
| --------------------------------- | ------------------ | ---------- |
| **LightGCN**                      | CF Baseline        | SIGIR 2020 |
| **SimGCL**                        | CF + CL Baseline   | SIGIR 2022 |
| **KGAT**                          | KG Baseline        | KDD 2019   |
| **KGCL**                          | KG + CL Baseline   | SIGIR 2022 |
| **KG-LightGCN** _(Biến thể 1)_    | KG + BPR           | This work  |
| **KG-LightGCN-CL** _(Biến thể 2)_ | KG + Cross-view CL | This work  |

---

## Chạy với Docker Desktop (Khuyến nghị)

### Yêu cầu

- Docker Desktop ≥ 4.x (Windows/macOS/Linux)
- NVIDIA Container Toolkit (nếu dùng GPU)
- RAM ≥ 16GB, GPU RTX 3050 hoặc tương đương

### Cài NVIDIA Container Toolkit (1 lần, nếu chưa có)

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Build image (1 lần)

```bash
docker compose build
# Hoặc:
docker build -t kg-lightgcn:latest .
```

### Kiểm tra môi trường

```bash
docker compose run --rm check
# Kết quả mong đợi: PyTorch OK, CUDA available, All models import OK
```

### Chuẩn bị dữ liệu

```bash
docker compose run --rm preprocess
# Tạo ra: data/processed/amazon-book/, data/processed/yelp2018/
# Cold splits: cold_10/, cold_20/, cold_30/
```

### Train từng model

```bash
# 1 seed (test nhanh ~3-5h/model trên RTX 3050)
docker compose run --rm train-lightgcn
docker compose run --rm train-simgcl
docker compose run --rm train-kgat
docker compose run --rm train-kgcl
docker compose run --rm train-kg-lightgcn
docker compose run --rm train-kg-lightgcn-cl

# Tất cả model, 5 seeds (đầy đủ)
docker compose run --rm train-all
```

### Các lệnh khác

```bash
# Ablation entity type (A1 none / A2 category / A3 brand / A4 full)
docker compose run --rm ablation

# Sensitivity analysis K và d
docker compose run --rm sensitivity

# Tổng hợp kết quả → CSV/LaTeX
docker compose run --rm aggregate

# Interactive shell để debug
docker compose run --rm shell
```

### Resume sau khi crash

Trainer tự động lưu checkpoint sau mỗi eval epoch. Chạy lại đúng lệnh cũ là tự resume:

```bash
# Lần đầu (bị crash ở epoch 135)
docker compose run --rm train-kgcl

# Chạy lại — tự động resume từ epoch 135
docker compose run --rm train-kgcl
# >> RESUMED from checkpoint: epoch=135, best_recall@20=0.0412
```

---

## Chạy không dùng Docker

### Cài đặt

```bash
git clone https://github.com/TranDucLongAIDHSPKTHY/KG-LightGCN_ICTA2026_v4.git
cd KG-LightGCN_ICTA2026_v4

# GPU CUDA 11.8
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Chạy

```bash
# Preprocessing
python3 scripts/download_data.py --dataset all
python3 scripts/preprocess.py --dataset all
python3 scripts/build_cold_split.py --dataset all --ratio 10 20 30

# Train
# Thay tên model: 'kg_lightgcn', 'kg_lightgcn_cl', 'kgat', 'kgcl', 'lightgcn', 'simgcl','''.
#Train 1 model/ 1 seeds
python3 main.py --model kg_lightgcn_cl --dataset amazon-book --seeds 42

# Train mỗi model 5 seeds
python3 main.py --model all --dataset amazon-book --seeds 42 0 1 2 3
```

#Train Cold-satrt
'''
python main.py --model kg_lightgcn --dataset amazon-book --cold_split cold_10 --seeds 42

## Note: Local: python3 -> python

## Cấu trúc thư mục

```
KG-LightGCN_ICTA2026/
├── Dockerfile                    ← Image definition
├── docker-compose.yml            ← Service definitions (train/eval/ablation)
├── requirements.txt
├── configs/
│   ├── base.yaml                 ← Fairness protocol (dim=64, lr=0.001, τ=0.2)
│   ├── fairness.yaml
│   └── model/                   ← Per-model hyperparameters
├── data/processed/
│   ├── amazon-book/              ← train/val/test + kg_full/category/brand
│   │   ├── cold_10/20/30/        ← Cold-start splits (seed=42)
│   │   └── stats.json, kg_meta.json
│   └── yelp2018/
├── models/
│   ├── lightgcn.py               ← He et al. SIGIR 2020
│   ├── simgcl.py                 ← Yu et al. SIGIR 2022
│   ├── kgat.py                   ← Wang et al. KDD 2019
│   ├── kgcl.py                   ← Yang et al. SIGIR 2022
│   └── kg_lightgcn.py            ← KGLightGCN + KGLightGCNCL (This work)
├── trainers/
│   ├── trainer.py                ← Base trainer + checkpoint/resume
│   └── kg_trainer.py             ← KG trainer (KGAT/KGCL/KG-LightGCN)
├── evaluation/
│   ├── full_ranking.py           ← Full-item ranking (không phải sampled)
│   ├── cold_evaluator.py         ← Cold-start metrics
│   └── stat_test.py              ← Paired t-test + Cohen's d
├── scripts/
│   ├── run_ablation.sh           ← 4 entity type variants
│   └── run_sensitivity.py        ← K, d, λ sensitivity
└── results/                      ← Sinh ra khi train
    ├── checkpoints/              ← Model weights (best + resume)
    ├── logs/                     ← Per-epoch metrics CSV
    └── tables/                   ← Aggregated results
```

---

## Tối ưu GPU (RTX 3050 — 4GB VRAM)

Project đã được tối ưu cho môi trường hạn chế VRAM:

| Vấn đề                              | Giải pháp                        | Tiết kiệm            |
| ----------------------------------- | -------------------------------- | -------------------- |
| `layer_out` list giữ K+1 tensor     | Running mean accumulation        | ~60% VRAM            |
| KGAT recompute entity_emb/batch     | Cache 1 lần/epoch, `detach()`    | ~99% redundant ops   |
| KGCL tạo augmented adj/batch        | Cache 1 lần/epoch                | ~1000x ít tensor tạm |
| Embedding sau eval không giải phóng | `del + empty_cache()`            | VRAM freed sau eval  |
| SimGCL propagate 3 lần/batch        | Clean propagation dùng chung BPR | 1 lần thay 3 lần     |

---

## Kết quả & Metrics

| Metric      | Mô tả                                 |
| ----------- | ------------------------------------- |
| `Recall@20` | Primary metric                        |
| `NDCG@20`   | Normalised Discounted Cumulative Gain |
| `HR@10`     | Hit Rate                              |
| `*_cold`    | Cold-start variants                   |

Xem log real-time:

```bash
tail -f results/logs/amazon-book/kgcl/seed42_run.log
```

---

## Tham khảo

- LightGCN: https://github.com/gusye1234/LightGCN-PyTorch
- SimGCL: https://github.com/Coder-Yu/QRec
- KGAT: https://github.com/xiangwang1223/knowledge_graph_attention_network
- KGCL: https://github.com/yuh-yang/KGCL-SIGIR22

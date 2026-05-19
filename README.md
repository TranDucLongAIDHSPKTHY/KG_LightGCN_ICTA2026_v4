# KG-LightGCN ICTA2026

> **Câu hỏi trung tâm:** Trong Collaborative Filtering, khi dùng cùng backbone LightGCN và InfoNCE loss, dạng structured side information nào tạo contrastive signal hiệu quả hơn: flat category, taxonomy, hay knowledge graph?

Đây là codebase chính thức cho bài báo nộp ICTA2026, nghiên cứu việc tích hợp Knowledge Graph vào LightGCN theo hai hướng: enrichment thuần (Biến thể 1) và cross-view contrastive learning (Biến thể 2).

---

## Mục lục

1. [Tổng quan kiến trúc](#tổng-quan-kiến-trúc)
2. [Các mô hình](#các-mô-hình)
3. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
4. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
5. [Cài đặt](#cài-đặt)
6. [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
7. [Cấu hình (Config)](#cấu-hình-config)
8. [Chạy thực nghiệm](#chạy-thực-nghiệm)
9. [Docker (Khuyến nghị)](#docker-khuyến-nghị)
10. [Kết quả & Metrics](#kết-quả--metrics)
11. [Ablation & Sensitivity Analysis](#ablation--sensitivity-analysis)
12. [Cold-start Evaluation](#cold-start-evaluation)
13. [Tối ưu GPU](#tối-ưu-gpu)
14. [Tham khảo](#tham-khảo)

---

## Tổng quan kiến trúc

### Sơ đồ model & data flow

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                        DỮ LIỆU ĐẦU VÀO                         │
  │                                                                 │
  │   Amazon-Book (KG + CF)          Yelp2018 (CF only)            │
  │   ├── train/val/test.txt         ├── train/val/test.txt        │
  │   ├── kg_full.txt                └── (no KG)                   │
  │   └── item2entity.json                                          │
  └──────────────────────┬──────────────────┬──────────────────────┘
                         │                  │
              ┌──────────▼──────────┐  ┌────▼──────────────┐
              │  KGDataset          │  │  CFDataset         │
              │  • KG triples       │  │  • norm_adj (Â)    │
              │  • entity mapping   │  │  • BPR sampling    │
              │  • KG sparse adj    │  │  • pickle-safe     │
              └──────────┬──────────┘  └────┬───────────────┘
                         │                  │
         ┌───────────────┘                  └──────────────────┐
         │  KGTrainer (alternating CF+KG)     Trainer (BPR/CL) │
         │  • _kgat_step()                    • _train_one_epoch│
         │  • _kgcl_step()                    • SimGCL branch  │
         │  • _kg_lightgcn_step()             • early stopping │
         │  • _kg_lightgcn_cl_step()          • resume ckpt    │
         └──────────────┬───────────────────────────┬──────────┘
                        │                           │
     ┌──────────────────┼──────────┐     ┌──────────┼──────────────┐
     │  KG Models       │          │     │  CF Models│             │
     │                  │          │     │           │             │
  ┌──▼──┐  ┌──────▼──┐  ┌───▼────┐ │  ┌──▼──────┐  ┌──▼──────┐   │
  │KGAT │  │  KGCL   │  │KGLight-│ │  │LightGCN │  │ SimGCL  │   │
  │Trans│  │KG-aug   │  │ GCN    │ │  │ (BPR)   │  │(BPR +   │   │
  │  R  │  │   CL    │  │ v1 & v2│ │  │         │  │InfoNCE) │   │
  └─────┘  └─────────┘  └────────┘ │  └─────────┘  └─────────┘   │
     │          │             │     │       │              │        │
     └──────────┴─────────────┴─────┴───────┴──────────────┘        │
                              │                                      │
                    ┌─────────▼──────────┐                          │
                    │     Evaluator      │◄─────────────────────────┘
                    │  full_ranking_eval │
                    │  • Recall@K        │
                    │  • NDCG@K          │
                    │  • HR@K            │
                    │  ColdEvaluator     │
                    │  • *_cold metrics  │
                    └─────────┬──────────┘
                              │
                   results/tables/*.json
                   results/logs/*/epoch_metrics.tsv
```

### Pipeline hoàn chỉnh

```
┌─────────────────────────────────────────────────────────────────────┐
│                       NGUỒN DỮ LIỆU THÔ                            │
│                                                                     │
│  Amazon-Book (CF interactions)                                      │
│  └─ github.com/gusye1234/LightGCN-PyTorch/data/amazon-book         │
│     train.txt, test.txt, item_list.txt, user_list.txt               │
│                                                                     │
│  Amazon-Book (KG metadata)                                          │
│  └─ [Primary]  jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/        │
│                meta_Books.json.gz  (Amazon 2018, có brand/publisher)│
│  └─ [Fallback] mcauleylab.ucsd.edu/public_datasets/data/amazon/    │
│                categoryFiles/meta_Books.json.gz  (Amazon 2014)      │
│                                                                     │
│  Yelp2018                                                           │
│  └─ github.com/Coder-Yu/QRec/dataset/yelp2018                      │
│     train.txt, test.txt                                             │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                   download_data.py
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      preprocess.py                                  │
│  1. 5-core filtering (user & item ≥ 5 interactions)                │
│  2. ID remap → [0, N) / [0, M)                                     │
│  3. Val split: 10% cuối train (shuffle seed=42)                    │
│  4. Build KG từ meta_Books.json.gz                                  │
│     (4 fwd relations + 4 inverse = 8 total)                        │
│  5. Reproducibility check (fingerprint MD5 × 3 runs)               │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                   build_cold_split.py
                   (cold_10 / cold_20 / cold_30)
                            │
                            ▼
                    data/processed/
                    amazon-book/  &  yelp2018/
                            │
                   main.py --model <name>
                            │
            ┌───────────────▼────────────────┐
            │      Training loop             │
            │  • eval mỗi 5 epoch            │
            │  • early stopping (patience=5) │
            │  • checkpoint: best + resume   │
            │  • ETA logging                 │
            └───────────────┬────────────────┘
                            │
            ┌───────────────▼────────────────┐
            │   Full-ranking evaluation      │
            │   Recall@20, NDCG@20,          │
            │   HR@10, NDCG@10               │
            │   + Cold-start (*_cold)        │
            └───────────────┬────────────────┘
                            │
                results/tables/<model>_results.json
                results/logs/<dataset>/<model>/seed*/
```

---

## Các mô hình

| Model                             | Loại               | Venue      | Loss                     | Mô tả                               |
| --------------------------------- | ------------------ | ---------- | ------------------------ | ----------------------------------- |
| **LightGCN**                      | CF Baseline        | SIGIR 2020 | BPR                      | Graph CF thuần, không có side info  |
| **SimGCL**                        | CF + CL Baseline   | SIGIR 2022 | BPR + InfoNCE            | Noise perturbation làm augmentation |
| **KGAT**                          | KG Baseline        | KDD 2019   | BPR + TransR             | Attention aggregation trên KG       |
| **KGCL**                          | KG + CL Baseline   | SIGIR 2022 | BPR + InfoNCE            | KG-guided edge dropout làm CL       |
| **KG-LightGCN** _(Biến thể 1)_    | KG + BPR           | This work  | BPR + KG Align           | KG enrichment item emb + BPR thuần  |
| **KG-LightGCN-CL** _(Biến thể 2)_ | KG + Cross-view CL | This work  | BPR + InfoNCE + KG Align | KG làm view cho cross-view CL       |

### Chi tiết KG-LightGCN (Biến thể 1 — Backbone)

```
1. KG entity propagation:
   E_entity = mean-pool(E^0, E^1, ..., E^kg_layers)  trên entity–entity graph

2. Item enrichment:
   enriched_i = σ(α) · e_i + (1 − σ(α)) · entity_i
   (α là learnable parameter)

3. LightGCN CF propagation:
   [user_final, item_final] = LightGCN(norm_adj, enriched_items)

4. Loss:
   L = BPR(u, i+, i−) + λ_kg · KGAlign + λ_reg · ||E||²
   KGAlign = mean(1 − cosine(item_emb, entity_emb))
```

### Chi tiết KG-LightGCN-CL (Biến thể 2 — Proposed)

```
Hai view bổ sung nhau:
  View CF  (perturbed): LightGCN + uniform noise ε tại mỗi layer, plain item emb
  View KG  (enriched):  LightGCN clean, KG-enriched item emb

Cross-view InfoNCE kéo cùng user/item gần nhau qua hai view.

Loss:
  L = BPR(u, i+, i−)
    + λ_cl · InfoNCE(user_CF, user_KG)     # user-level CL
    + λ_cl · InfoNCE(item_CF, item_KG)     # item-level CL
    + λ_kg · KGAlign                        # entity ↔ item alignment
    + λ_reg · ||E||²                        # L2 regularisation

Khi kg_type='none': View KG = noise view thứ hai → tương đương SimGCL
(dùng cho so sánh ablation)
```

---

## Cấu trúc thư mục

```
KG-LightGCN_ICTA2026/
│
├── main.py                        ← Entry point thống nhất cho mọi model
├── Dockerfile                     ← GPU (CUDA 11.8) + CPU build
├── docker-compose.yml             ← Services: train/eval/ablation/sensitivity
├── requirements.txt               ← Python deps (numpy, scipy, pyyaml, tqdm)
│
├── configs/
│   ├── base.yaml                  ← Fairness protocol chung (lr, dim, batch_size)
│   ├── fairness.yaml              ← Các ràng buộc fairness (embedding_dim=64 HARD)
│   └── model/
│       ├── lightgcn.yaml          ← LightGCN hyperparams (weight_decay=1e-4)
│       ├── simgcl.yaml            ← SimGCL (weight_decay=0, eps=0.1, λ=0.5)
│       ├── kgat.yaml              ← KGAT (lr=0.0001, weight_decay=1e-5)
│       ├── kgcl.yaml              ← KGCL (λ_cl=0.1, early_stopping=5)
│       ├── kg_lightgcn.yaml       ← KG-LightGCN Biến thể 1
│       └── kg_lightgcn_cl.yaml    ← KG-LightGCN-CL Biến thể 2
│
├── models/
│   ├── base_model.py              ← Abstract BaseModel (forward, get_embeddings, predict)
│   ├── lightgcn.py                ← LightGCN + lazy device move
│   ├── simgcl.py                  ← SimGCL (BUG-1/2/3 fixed)
│   ├── kgat.py                    ← KGAT + OOM fix (chunked entity emb)
│   ├── kgcl.py                    ← KGCL (BUG-1/2/3 fixed)
│   └── kg_lightgcn.py             ← KGLightGCN + KGLightGCNCL (_KGEnrichMixin)
│
├── trainers/
│   ├── trainer.py                 ← Base Trainer: BPR, SimGCL, early stopping, resume
│   └── kg_trainer.py              ← KGTrainer: KGAT/KGCL/KG-LightGCN step functions
│
├── datasets/
│   ├── base_dataset.py            ← Abstract BaseDataset (I/O, negative sampling)
│   ├── cf_dataset.py              ← CFDataset: norm_adj, BPR sampling, pickle-safe
│   ├── kg_dataset.py              ← KGDataset: KG triples, entity mapping, adj
│   └── dataloader.py              ← Factory: get_cf_dataloader, get_kg_dataloader
│
├── losses/
│   ├── bpr_loss.py                ← BPR loss + KG BPR (TransR)
│   └── contrastive_loss.py        ← InfoNCE / NT-Xent loss
│
├── evaluation/
│   ├── metrics.py                 ← Recall@K, NDCG@K, HR@K (vectorised)
│   ├── full_ranking.py            ← Full-item ranking eval (tối ưu batch)
│   ├── evaluator.py               ← Unified Evaluator (val/test)
│   ├── cold_evaluator.py          ← Cold-start evaluation
│   └── stat_test.py               ← Paired t-test, Cohen's d
│
├── utils/
│   ├── config.py                  ← YAML loader + .env + CLI override
│   ├── logger.py                  ← Module + run + epoch + summary loggers
│   └── seed.py                    ← set_seed (all frameworks), get_seeds=[42,0,1,2,3]
│
├── scripts/
│   ├── download_data.py           ← Tải Amazon-Book (LightGCN repo) + Yelp2018
│   ├── preprocess.py              ← 5-core filter, remap, val split, build KG
│   ├── build_cold_split.py        ← Tạo cold-start splits (10/20/30%)
│   ├── run_train.py               ← Train wrapper cho SLURM/batch jobs
│   ├── run_ablation.sh            ← Entity ablation (none/category/brand/full)
│   ├── run_sensitivity.py         ← Sensitivity: K, d, λ
│   ├── run_significance.py        ← Paired t-test giữa các model
│   ├── run_eval.py                ← Eval từ checkpoint đã lưu
│   └── run_all.sh                 ← Full pipeline: preprocess → train → eval → aggregate
│
├── results/
│   └── aggregate.py               ← Tổng hợp JSON → CSV + LaTeX table
│
└── data/processed/                ← Sinh ra sau preprocess.py (không push git)
    ├── amazon-book/
    │   ├── train.txt / val.txt / test.txt
    │   ├── kg_full.txt / kg_category.txt / kg_brand.txt
    │   ├── item2entity.json / id_maps.json
    │   ├── stats.json / kg_meta.json
    │   └── cold_10/ cold_20/ cold_30/
    └── yelp2018/
        ├── train.txt / val.txt / test.txt
        └── stats.json
```

---

## Yêu cầu hệ thống

| Thành phần | Tối thiểu       | Khuyến nghị     |
| ---------- | --------------- | --------------- |
| GPU VRAM   | 4 GB (RTX 3050) | 8–16 GB         |
| RAM        | 16 GB           | 32 GB           |
| Ổ cứng     | 50 GB (HDD ok)  | SSD cho results |
| CUDA       | 11.8+           | 11.8            |
| Python     | 3.10            | 3.10            |
| PyTorch    | 2.1.2           | 2.1.2           |

> **Lưu ý:** Project đã được tối ưu đặc biệt cho RTX 3050 (4 GB VRAM). Xem phần [Tối ưu GPU](#tối-ưu-gpu) để biết chi tiết.

---

## Cài đặt

### Cách 1: Môi trường local

```bash
git clone https://github.com/TranDucLongAIDHSPKTHY/KG-LightGCN_ICTA2026_v4.git
cd KG-LightGCN_ICTA2026_v4

# GPU (CUDA 11.8)
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Các thư viện còn lại
pip install -r requirements.txt
```

### Cách 2: Docker (khuyến nghị — xem phần [Docker](#docker-khuyến-nghị))

### Cấu hình .env

Tạo file `.env` từ mẫu:

```bash
cp .env.example .env
```

Chỉnh sửa theo môi trường của bạn:

```env
# Thư mục chứa dataset (trên HDD)
DATA_ROOT=/hdd/datasets

# Thư mục lưu kết quả: logs, checkpoints, tables (nên dùng SSD)
RESULTS_ROOT=/ssd/kg-lightgcn/results

# Device: auto | cpu | cuda | cuda:0
DEVICE=auto

# Số DataLoader workers
NUM_WORKERS=4
```

Nếu không tạo `.env`, hệ thống dùng đường dẫn mặc định: `./data/processed` và `./results`.

---

## Chuẩn bị dữ liệu

### Bước 1: Tải dữ liệu thô

```bash
# Tải tất cả
python scripts/download_data.py --dataset all --raw_dir /path/to/raw

# Chỉ Amazon-Book
python scripts/download_data.py --dataset amazon-book --raw_dir /path/to/raw

# Chỉ kiểm tra (không tải)
python scripts/download_data.py --dataset all --check_only
```

> **Quan trọng về `meta_Books.json.gz`:**
> Script sẽ thử tải tự động từ nhiều mirror (Amazon 2018 + 2014).
> Nếu thất bại, tải thủ công tại: https://nijianmo.github.io/amazon/ → Books → metadata
> (cần điền Google Form). Đặt file vào `data/raw/amazon-book/meta_Books.json.gz`.

### Bước 2: Preprocessing

```bash
python scripts/preprocess.py --dataset all \
    --raw_dir /path/to/raw \
    --out_dir /path/to/processed
```

**Pipeline xử lý (Amazon-Book):**

1. Đọc `train.txt`, `test.txt` (định dạng LightGCN)
2. 5-core filtering (user và item phải xuất hiện ≥ 5 lần)
3. Remap ID → `[0, N)` / `[0, M)`
4. Tách val từ train pool (10% cuối sau shuffle, seed=42)
5. Xây KG từ `meta_Books.json.gz`:
   - Relation 0: `also_bought` (item→item)
   - Relation 1: `also_viewed` (item→item)
   - Relation 2: `has_category` (item→category)
   - Relation 3: `has_brand` (item→brand)
   - Relation 4–7: inverse của 0–3
6. Kiểm tra reproducibility (3 lần, fingerprint MD5)
7. Lưu files + stats

**Output files:**

```
data/processed/amazon-book/
├── train.txt           # user_id item1 item2 ...
├── val.txt
├── test.txt
├── kg_full.txt         # h<TAB>r<TAB>t (8 relations, fwd+inv)
├── kg_category.txt     # chỉ relations 2, 6
├── kg_brand.txt        # chỉ relations 3, 7
├── item2entity.json    # {item_id: entity_id}
├── id_maps.json        # user_map, item_map
├── stats.json          # n_users, n_items, density, ...
└── kg_meta.json        # n_entities, n_relations, entity_ranges
```

### Bước 3: Tạo cold-start splits

```bash
python scripts/build_cold_split.py \
    --dataset all \
    --ratio 10 20 30 \
    --data_dir /path/to/processed
```

Tạo ra `cold_10/`, `cold_20/`, `cold_30/` trong mỗi dataset. Mỗi split chứa:

- `train.txt`, `val.txt`: đã loại bỏ cold items
- `test.txt`: giữ nguyên (để evaluate)
- `kg_full.txt`: chỉ giữ triples liên quan đến cold items
- `cold_items.txt`: danh sách cold item IDs
- `cold_stats.json`: thống kê split

---

## Cấu hình (Config)

Hệ thống config theo thứ tự ưu tiên (cao → thấp):

```
CLI --override  >  Model YAML  >  Base YAML  >  .env  >  Default
```

### `configs/base.yaml` — Fairness Protocol

```yaml
model:
  embedding_dim: 64 # HARD CONSTRAINT — không đổi per-model

train:
  epochs: 1000
  batch_size: 2048
  learning_rate: 0.001
  weight_decay: 1.0e-4
  early_stopping_patience: 5
  early_stopping_metric: recall@20

eval:
  eval_interval: 5 # evaluate mỗi 5 epoch
  top_k: [10, 20]

contrastive:
  temperature: 0.2
  lambda_cl: 0.5
```

### Per-model YAML overrides

| Model            | Key overrides                                          |
| ---------------- | ------------------------------------------------------ |
| `lightgcn`       | `weight_decay=1e-4`, `early_stopping=10`               |
| `simgcl`         | `weight_decay=0.0`, `eps=0.1`, `lambda_cl=0.5`         |
| `kgat`           | `lr=0.0001`, `weight_decay=1e-5`, `kg_batch_size=2048` |
| `kgcl`           | `lambda_cl=0.1`, `lambda_kg=0.1`, `early_stopping=5`   |
| `kg_lightgcn`    | `kg_n_layers=2`, `kg_reg=1e-5`, `entity_agg=mean`      |
| `kg_lightgcn_cl` | `eps=0.1`, `temperature=0.2`, `lambda_cl=0.5`          |

---

## Chạy thực nghiệm

### Train một model

```bash
# LightGCN — baseline
python main.py --model lightgcn --dataset amazon-book --seeds 42

# KG-LightGCN (Biến thể 1)
python main.py --model kg_lightgcn --dataset amazon-book --seeds 42

# KG-LightGCN-CL (Biến thể 2 — proposed)
python main.py --model kg_lightgcn_cl --dataset amazon-book --seeds 42

# Yelp2018 (chỉ CF models)
python main.py --model lightgcn --dataset yelp2018 --seeds 42
python main.py --model simgcl --dataset yelp2018 --seeds 42
```

### Train multi-seed (đầy đủ, 5 seeds)

```bash
python main.py --model all --dataset amazon-book --seeds 42 0 1 2 3
```

### Cold-start evaluation

```bash
# Train trên cold_20 split
python main.py --model kg_lightgcn --dataset amazon-book \
    --cold_split cold_20 --seeds 42

# Các mức cold
python main.py --model lightgcn --dataset amazon-book --cold_split cold_10 --seeds 42
python main.py --model lightgcn --dataset amazon-book --cold_split cold_30 --seeds 42
```

### Override hyperparameters

```bash
# Đổi số layers
python main.py --model lightgcn --dataset amazon-book --n_layers 4

# Đổi weight decay
python main.py --model kg_lightgcn --dataset amazon-book --weight_decay 1e-3

# Ablation: chỉ dùng category KG
python main.py --model kg_lightgcn --dataset amazon-book --kg_type category

# Override bất kỳ param (dạng key=value)
python main.py --model lightgcn --dataset amazon-book \
    --override train.learning_rate=0.0005 model.n_layers=4
```

### Eval từ checkpoint đã lưu

```bash
python scripts/run_eval.py \
    --model kg_lightgcn \
    --dataset amazon-book \
    --checkpoint results/checkpoints/amazon-book/kg_lightgcn/seed42_best.pt \
    --split test

# Kèm cold eval
python scripts/run_eval.py \
    --model kg_lightgcn \
    --checkpoint results/checkpoints/amazon-book/kg_lightgcn/seed42_best.pt \
    --cold_dir data/processed/amazon-book/cold_20
```

### Tổng hợp kết quả

```bash
python results/aggregate.py --dataset amazon-book
# Xuất ra:
#   results/tables/main_results_amazon-book.csv
#   results/tables/main_results_amazon-book.tex
```

### Kiểm định thống kê

```bash
python scripts/run_significance.py --dataset amazon-book
# Thực hiện paired t-test cho các cặp:
#   KG-LightGCN vs LightGCN
#   KG-LightGCN vs KGCL
#   KG-LightGCN-CL vs LightGCN
#   KG-LightGCN-CL vs KGCL
#   KG-LightGCN-CL vs KG-LightGCN
```

### Resume sau khi crash

Trainer tự động lưu checkpoint sau mỗi lần evaluate. Chạy lại **đúng lệnh cũ** sẽ tự resume:

```bash
# Lần đầu (bị crash ở epoch 135)
python main.py --model kgcl --dataset amazon-book --seeds 42

# Chạy lại → tự resume từ epoch 135
python main.py --model kgcl --dataset amazon-book --seeds 42
# >> RESUMED from checkpoint: epoch=135, best_recall@20=0.0412
```

Checkpoint được lưu tại:

```
results/checkpoints/<dataset>/<model>/
├── seed42_best.pt      # model tốt nhất (dùng cho eval)
└── seed42_resume.pt    # checkpoint mới nhất (dùng để resume)
```

---

## Docker (Khuyến nghị)

### Yêu cầu

- Docker Desktop ≥ 4.x
- NVIDIA Container Toolkit (nếu dùng GPU)

### Cài NVIDIA Container Toolkit (một lần)

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Build image

```bash
# GPU
docker compose build   # hoặc: docker build -t kg-lightgcn:gpu .

# CPU only
docker build --build-arg MODE=cpu -t kg-lightgcn:cpu .
```

### Kiểm tra môi trường

```bash
docker compose --profile gpu run --rm check-gpu
docker compose --profile cpu run --rm check-cpu
```

### Preprocessing (chạy một lần)

```bash
docker compose --profile cpu run --rm preprocess
```

### Train models

```bash
# GPU — từng model
docker compose --profile gpu run train-lightgcn
docker compose --profile gpu run train-simgcl
docker compose --profile gpu run train-kgat
docker compose --profile gpu run train-kgcl
docker compose --profile gpu run train-kg-lightgcn
docker compose --profile gpu run train-kg-lightgcn-cl

# GPU — tất cả models, 5 seeds
docker compose --profile gpu run train-all-gpu

# CPU (fallback)
docker compose --profile cpu run train-lightgcn-cpu
docker compose --profile cpu run train-all-cpu
```

### Ablation & Sensitivity

```bash
docker compose --profile gpu run ablation
docker compose --profile gpu run sensitivity
```

### Tổng hợp kết quả

```bash
docker compose --profile cpu run --rm aggregate
```

### Interactive shell (debug)

```bash
docker compose --profile gpu run shell-gpu
docker compose --profile cpu run shell-cpu
```

### Xem log real-time

```bash
# Xem log trong container đang chạy
docker compose logs -f train-kgcl

# Hoặc theo dõi file log trực tiếp (nếu RESULTS_ROOT được mount)
tail -f $RESULTS_ROOT/logs/amazon-book/kgcl/seed42/train.log
```

---

## Kết quả & Metrics

### Metrics được tính

| Metric      | Mô tả                                                  |
| ----------- | ------------------------------------------------------ |
| `Recall@20` | **Primary metric** — tỷ lệ items relevant trong top-20 |
| `NDCG@20`   | Normalized Discounted Cumulative Gain tại K=20         |
| `HR@10`     | Hit Rate — có ít nhất 1 item relevant trong top-10     |
| `NDCG@10`   | NDCG tại K=10                                          |
| `*_cold`    | Các metrics trên với chỉ cold items                    |

> **Lưu ý:** Tất cả metrics dùng **full-item ranking** (không phải sampled ranking), toàn bộ ~90K items được xếp hạng cho mỗi user.

### Cấu trúc log

```
results/
├── logs/
│   └── amazon-book/
│       └── kg_lightgcn_cl/
│           └── seed42/
│               ├── train.log          ← toàn bộ quá trình train
│               ├── epoch_metrics.tsv  ← epoch | loss | recall@20 | ...
│               └── summary.json       ← kết quả cuối (best_epoch, test_metrics)
├── checkpoints/
│   └── amazon-book/
│       └── kg_lightgcn_cl/
│           ├── seed42_best.pt
│           └── seed42_resume.pt
└── tables/
    ├── kg_lightgcn_results.json
    ├── main_results_amazon-book.csv
    ├── main_results_amazon-book.tex
    └── significance_kg_lightgcn_cl_vs_lightgcn.json
```

### Format kết quả JSON

```json
{
  "model": "kg_lightgcn_cl",
  "dataset": "amazon-book",
  "mean": {
    "recall@20": 0.0456,
    "ndcg@20": 0.0312,
    "hr@10": 0.0678,
    "ndcg@10": 0.0289
  },
  "std": { "recall@20": 0.0012, ... },
  "per_seed": [
    { "seed": 42, "best_epoch": 87, "test_metrics": {...} },
    ...
  ]
}
```

---

## Ablation & Sensitivity Analysis

### Entity Ablation (A1–A4)

Kiểm tra đóng góp của từng loại KG entity:

```bash
bash scripts/run_ablation.sh
```

| Setting | kg_type    | Mô tả                                                |
| ------- | ---------- | ---------------------------------------------------- |
| A1      | `none`     | Không có KG (giảm về LightGCN)                       |
| A2      | `category` | Chỉ category entities                                |
| A3      | `brand`    | Chỉ brand/publisher entities                         |
| A4      | `full`     | Tất cả: also_bought + also_viewed + category + brand |

Hoặc chạy thủ công:

```bash
python main.py --model kg_lightgcn --dataset amazon-book --kg_type none --seeds 42 0 1 2 3
python main.py --model kg_lightgcn --dataset amazon-book --kg_type category --seeds 42 0 1 2 3
python main.py --model kg_lightgcn --dataset amazon-book --kg_type brand --seeds 42 0 1 2 3
python main.py --model kg_lightgcn --dataset amazon-book --kg_type full --seeds 42 0 1 2 3
```

### Sensitivity Analysis

```bash
# Tất cả params
python scripts/run_sensitivity.py --dataset amazon-book --param all

# Từng param riêng
python scripts/run_sensitivity.py --dataset amazon-book --param n_layers
python scripts/run_sensitivity.py --dataset amazon-book --param embedding_dim
python scripts/run_sensitivity.py --dataset amazon-book --param weight_decay
```

| Parameter           | Grid               |
| ------------------- | ------------------ |
| `n_layers` (K)      | {1, 2, 3, 4}       |
| `embedding_dim` (d) | {32, 64, 128, 256} |
| `weight_decay` (λ)  | {0.01, 0.1, 1.0}   |

---

## Cold-start Evaluation

**Protocol Cold-20:** 20% items ít tương tác nhất được chọn làm "cold items".

- Train/val: loại bỏ interactions với cold items
- Test: giữ nguyên (để evaluate trên cold items)
- KG: giữ triples có ít nhất một đầu là cold item

```bash
# Train trên cold_20
python main.py --model kg_lightgcn --dataset amazon-book \
    --cold_split cold_20 --seeds 42

# Eval tự động chạy sau khi train xong (lightgcn, kg_lightgcn, kg_lightgcn_cl)
# Kết quả lưu tại: results/tables/<model>_cold20_metrics.json
```

Cold metrics được đặt tên với suffix `_cold`:

- `recall@20_cold`, `ndcg@20_cold`, `hr@10_cold`, `ndcg@10_cold`

---

## Tối ưu GPU

Project đã tối ưu nhiều vấn đề hiệu năng quan trọng, đặc biệt cho GPU 4 GB VRAM:

| Vấn đề                                                     | Giải pháp                                                   | Tiết kiệm                      |
| ---------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------ |
| `layer_out` list giữ K+1 tensors                           | Running mean accumulation                                   | ~60% VRAM                      |
| KGAT recompute entity_emb mỗi batch (1.4M triples × 21 GB) | Cache 1 lần/epoch + chunked projection (`chunk_size=32768`) | ~99% redundant ops             |
| KGCL tạo augmented adj mỗi batch                           | Cache `_aug_adj1`, `_aug_adj2` 1 lần/epoch                  | ~1000× ít tensor tạm           |
| Embedding sau eval không giải phóng                        | `del + gc.collect() + torch.cuda.empty_cache()`             | VRAM freed sau eval            |
| SimGCL propagate 3 lần/batch                               | Clean propagation dùng chung cho BPR                        | 1 lần thay 3 lần               |
| Sparse tensor không pickle được (Windows)                  | `__getstate__`/`__setstate__` trong `CFDataset`             | `num_workers > 0` trên Windows |
| KG tensors 1.4M triples trên GPU                           | Giữ CPU, move chunk-by-chunk                                | ~33 MB VRAM tiết kiệm constant |

### Điều chỉnh `chunk_size` cho KGAT

Nếu vẫn OOM với KGAT, giảm `chunk_size`:

```yaml
# configs/model/kgat.yaml
model_extra:
  chunk_size: 8192 # default: 32768; giảm nếu OOM trên GPU < 4 GB
```

---

## Fairness Protocol

Để đảm bảo so sánh công bằng giữa các model:

- **Embedding dim = 64** (HARD constraint, không override per-model)
- **Optimizer = Adam**, lr = 0.001 (trừ KGAT: 0.0001 theo paper gốc)
- **Batch size = 2048** cho tất cả
- **Negative sampling = 1** (uniform)
- **Full-item ranking** (không phải sampled)
- **5 seeds** = {42, 0, 1, 2, 3}, báo cáo mean ± std
- **Eval interval = 5 epochs** (không eval mỗi epoch)

---

## Tham khảo

### Papers

- **LightGCN:** He et al., SIGIR 2020 — [arxiv](https://arxiv.org/abs/2002.02126)
- **SimGCL:** Yu et al., SIGIR 2022 — [arxiv](https://arxiv.org/abs/2112.08679)
- **KGAT:** Wang et al., KDD 2019 — [arxiv](https://arxiv.org/abs/1905.07854)
- **KGCL:** Yang et al., SIGIR 2022 — [arxiv](https://arxiv.org/abs/2205.00976)
- **BPR:** Rendle et al., UAI 2009 — [arxiv](https://arxiv.org/abs/1205.2618)

### Official Implementations

- LightGCN-PyTorch: https://github.com/gusye1234/LightGCN-PyTorch
- SimGCL (QRec): https://github.com/Coder-Yu/QRec
- KGAT: https://github.com/xiangwang1223/knowledge_graph_attention_network
- KGCL: https://github.com/yuh-yang/KGCL-SIGIR22

### Dữ liệu

- Amazon-Book (CF): https://github.com/gusye1234/LightGCN-PyTorch/tree/master/data/amazon-book
- Amazon-Book (Meta/KG): https://nijianmo.github.io/amazon/
- Yelp2018: https://github.com/Coder-Yu/QRec/tree/master/dataset/yelp2018

---

## Bugs đã fix

Repo này đã sửa nhiều bugs so với implementation gốc:

| File                    | Bug                                                                              | Fix                                                         |
| ----------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `models/simgcl.py`      | Noise generation sai (normalize rồi scale = fixed magnitude, không phải uniform) | `noise = (rand * 2 - 1) * eps` per element, không normalize |
| `models/kgcl.py`        | `get_embeddings()`: `adj` undefined (NameError)                                  | Dùng `self.norm_adj`                                        |
| `models/kgcl.py`        | `_kg_propagation()`: relation emb bị ignore hoàn toàn                            | Incorporate `e_t ⊙ e_r` gate                                |
| `models/kgcl.py`        | `_augment_adj()`: `p_keep` formula sai → không có dropout                        | `p_keep = drop_prob + (1-drop_prob)*degree`                 |
| `models/kgat.py`        | `graphsage`: `W_gc` input dim sai (D thay vì 2D)                                 | `Linear(2*embedding_dim, embedding_dim)`                    |
| `models/kgat.py`        | CF propagation dùng mean-pool như LightGCN (sai paper KGAT)                      | Concat layer outputs → W_out projection                     |
| `models/kgat.py`        | OOM: `_project()` trên 1.4M triples = 21 GB                                      | Chunked projection + CPU tensors                            |
| `evaluation/metrics.py` | `Recall@K` chia `min(\|GT\|, k)` (non-standard, inflate)                         | Chia `\|GT\|` đúng chuẩn                                    |
| `trainers/trainer.py`   | `_run_cold_eval()`: checkpoint path sai → NameError                              | Sửa path `<dir>/<dataset>/<model>/seed42_best.pt`           |

---

_Tác giả: TranDucLong — AIDHSPKTHY_
_Phiên bản: v4 (ICTA2026 submission)_

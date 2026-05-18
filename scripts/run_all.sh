#!/usr/bin/env bash
# scripts/run_all.sh
# ─────────────────────────────────────────────────────────────────────────────
# Full experiment pipeline:
#   1. Preprocess datasets
#   2. Build cold splits
#   3. Train all models (multi-seed)
#   4. Run significance tests
#   5. Aggregate results
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DATASET="${1:-amazon-book}"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

echo "======================================================"
echo " KG-LightGCN Full Pipeline | Dataset: $DATASET"
echo "======================================================"

# Step 1: Preprocess
echo "[1/5] Preprocessing $DATASET ..."
python scripts/preprocess.py --dataset "$DATASET" 2>&1 | tee "$LOG_DIR/preprocess.log"

# Step 2: Cold splits
echo "[2/5] Building cold splits ..."
python scripts/build_cold_split.py --dataset "$DATASET" --ratio 10 20 30 \
    2>&1 | tee "$LOG_DIR/cold_splits.log"

# Step 3: Train all models (multi-seed)
echo "[3/5] Training all models ..."
MODELS="lightgcn kg_lightgcn "

# Only KG models on amazon-book
if [ "$DATASET" = "amazon-book" ]; then
    MODELS="lightgcn simgcl kgat kgcl kg_lightgcn kg_lightgcn_cl"
fi

for model in $MODELS; do
    echo "  Training $model ..."
    python main.py --model "$model" --dataset "$DATASET" \
        2>&1 | tee "$LOG_DIR/train_${model}.log"
done

# Step 4: Significance tests
echo "[4/5] Running significance tests ..."
python scripts/run_significance.py --dataset "$DATASET" \
    2>&1 | tee "$LOG_DIR/significance.log"

# Step 5: Aggregate results
echo "[5/5] Aggregating results ..."
python results/aggregate.py --dataset "$DATASET" \
    2>&1 | tee "$LOG_DIR/aggregate.log"

echo ""
echo "======================================================"
echo " Pipeline complete. Results in results/tables/"
echo "======================================================"

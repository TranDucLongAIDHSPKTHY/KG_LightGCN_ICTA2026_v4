#!/usr/bin/env bash
# scripts/run_ablation.sh
# ─────────────────────────────────────────────────────────────────────────────
# Entity ablation study for KG-LightGCN on Amazon-Book.
# Settings: A1 none, A2 category, A3 brand, A4 full
# Each setting runs 5 seeds; report mean ± std.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DATASET="amazon-book"
LOG_DIR="results/logs/ablation"
mkdir -p "$LOG_DIR"

echo "======================================================"
echo " Entity Ablation Study | KG-LightGCN | $DATASET"
echo "======================================================"

declare -A KG_TYPES
KG_TYPES["A1"]="none"
KG_TYPES["A2"]="category"
KG_TYPES["A3"]="brand"
KG_TYPES["A4"]="full"

for model_variant in kg_lightgcn kg_lightgcn_cl; do
    echo ""
    echo "===== $model_variant ====="
    for label in A1 A2 A3 A4; do
        kg_type="${KG_TYPES[$label]}"
        echo ""
        echo "--- $model_variant | $label: kg_type=$kg_type ---"
        python main.py \
            --model "$model_variant" \
            --dataset "$DATASET" \
            --kg_type "$kg_type" \
            --seeds 42 0 1 2 3 \
            2>&1 | tee "$LOG_DIR/ablation_${model_variant}_${label}_${kg_type}.log"
    done
done

echo ""
echo "======================================================"
echo " Ablation complete. Results in results/tables/"
echo "======================================================"

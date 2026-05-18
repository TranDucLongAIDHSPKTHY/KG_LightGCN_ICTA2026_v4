"""
results/aggregate.py
─────────────────────────────────────────────────────────────────────────────
Aggregate per-model result JSONs into comparison tables (CSV + LaTeX).

Outputs:
  results/tables/main_results_<dataset>.csv
  results/tables/main_results_<dataset>.tex
  results/tables/cold_results_<dataset>.csv

Usage:
  python results/aggregate.py --dataset amazon-book
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("aggregate")

ALL_MODELS = ["lightgcn", "simgcl", "kgat", "kgcl", "kg_lightgcn", "kg_lightgcn_cl"]
METRICS = ["recall@20", "ndcg@20", "hr@10", "ndcg@10"]


def load_result(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def format_mean_std(mean: float, std: float, bold: bool = False) -> str:
    s = f"{mean:.4f}±{std:.4f}"
    return f"\\textbf{{{s}}}" if bold else s


def build_main_table(dataset: str, result_dir: str) -> List[Dict]:
    rows = []
    for model in ALL_MODELS:
        path = os.path.join(result_dir, f"{model}_results.json")
        data = load_result(path)
        if data is None:
            continue
        row = {"model": model}
        for metric in METRICS:
            m = data.get("mean", {}).get(metric, None)
            s = data.get("std", {}).get(metric, None)
            row[metric] = (m, s)
        rows.append(row)
    return rows


def rows_to_csv(rows: List[Dict], metrics: List[str]) -> str:
    header = "model," + ",".join(f"{m}_mean,{m}_std" for m in metrics)
    lines = [header]
    for row in rows:
        parts = [row["model"]]
        for m in metrics:
            val = row.get(m)
            if val is not None and val[0] is not None:
                parts.extend([f"{val[0]:.6f}", f"{val[1]:.6f}"])
            else:
                parts.extend(["N/A", "N/A"])
        lines.append(",".join(parts))
    return "\n".join(lines)


def rows_to_latex(rows: List[Dict], metrics: List[str], dataset: str) -> str:
    """Generate a LaTeX booktabs table."""
    col_fmt = "l" + "c" * len(metrics)
    header_cols = " & ".join(m.replace("@", "@") for m in metrics)
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
        f"Model & {header_cols} \\\\",
        "\\midrule",
    ]

    # Find best metric values for bold formatting
    best = {}
    for m in metrics:
        vals = [r.get(m, (None, None))[0] for r in rows if r.get(m, (None,))[0] is not None]
        best[m] = max(vals) if vals else None

    for row in rows:
        parts = [row["model"].replace("_", "\\_")]
        for m in metrics:
            val = row.get(m)
            if val is not None and val[0] is not None:
                mean, std = val
                is_best = best.get(m) is not None and abs(mean - best[m]) < 1e-8
                parts.append(format_mean_std(mean, std, bold=is_best))
            else:
                parts.append("--")
        lines.append(" & ".join(parts) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Main results on {dataset}}}", 
        "\\end{table}",
    ]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="amazon-book")
    parser.add_argument("--result_dir", default="results/tables")
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = args.result_dir
    dataset = args.dataset
    os.makedirs(result_dir, exist_ok=True)

    rows = build_main_table(dataset, result_dir)
    if not rows:
        logger.warning(f"No result files found in {result_dir}. Nothing to aggregate.")
        return

    # CSV
    csv_str = rows_to_csv(rows, METRICS)
    csv_path = os.path.join(result_dir, f"main_results_{dataset}.csv")
    with open(csv_path, "w") as f:
        f.write(csv_str)
    logger.info(f"CSV table saved: {csv_path}")

    # LaTeX
    latex_str = rows_to_latex(rows, METRICS, dataset)
    tex_path = os.path.join(result_dir, f"main_results_{dataset}.tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)
    logger.info(f"LaTeX table saved: {tex_path}")

    # Print to console
    print("\n" + "=" * 70)
    print(f"MAIN RESULTS — {dataset}")
    print("=" * 70)
    header = f"{'Model':<18}" + "".join(f"{m:>20}" for m in METRICS)
    print(header)
    print("-" * 70)
    for row in rows:
        vals = []
        for m in METRICS:
            v = row.get(m)
            if v and v[0] is not None:
                vals.append(f"{v[0]:.4f}±{v[1]:.4f}")
            else:
                vals.append("N/A")
        print(f"{row['model']:<18}" + "".join(f"{v:>20}" for v in vals))
    print("=" * 70)

    # Cold results (if available)
    cold_rows = []
    for model in ["lightgcn", "kg_lightgcn", "kg_lightgcn_cl"]:
        path = os.path.join(result_dir, f"{model}_cold20_metrics.json")
        data = load_result(path)
        if data:
            cold_rows.append({"model": model, **data})

    if cold_rows:
        cold_metrics = [k for k in cold_rows[0] if k != "model"]
        cold_csv = rows_to_csv(
            [{"model": r["model"], **{m: (r.get(m), 0.0) for m in cold_metrics}} for r in cold_rows],
            cold_metrics,
        )
        cold_path = os.path.join(result_dir, f"cold_results_{dataset}.csv")
        with open(cold_path, "w") as f:
            f.write(cold_csv)
        logger.info(f"Cold results saved: {cold_path}")


if __name__ == "__main__":
    main()

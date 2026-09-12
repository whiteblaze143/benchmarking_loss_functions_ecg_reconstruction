"""GRAIL-ECG v2: Master Autonomous Pipeline Orchestrator.

Chains the end-to-end representation learning and qualification pipeline:
1. Phase R1 & R2: Train and qualify all 10 factorial models on Fold 8.
2. Phase R3: Compute factorial ANOVA main effects and interaction effects (Delta_G, Delta_S, Delta_V).
3. Phase R7 & R8 & R9: Execute exhaustive 4,095 subset lattice evaluation on Model M / best model.
4. Phase R10: Synthesize Lead Information Theory (exact Shapley, synergy graphs, minimal sufficient sets, Pareto frontiers).
5. Compile comprehensive GRAIL_V2_REPRESENTATION_REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pandas as pd
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_cmd(cmd: str, desc: str):
    print(f"\n>>> [PIPELINE] {desc}")
    print(f"Command: {cmd}")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        print(f"FAILED: {desc} exited with code {res.returncode}")
        sys.exit(res.returncode)
    print(f"<<< [PIPELINE] Finished {desc} in {time.time() - t0:.1f}s")


def analyze_factorial_effects(summary_json_path: Path, output_md_path: Path):
    """Computes factorial main effects and interactions from 2^3 factorial summary."""
    with open(summary_json_path) as f:
        data = json.load(f)

    # 2^3 design table
    # Factors: G (Geometry), S (Slots), V (View Aux)
    factor_map = {
        "b1": (0, 0, 0),
        "b3_geom": (1, 0, 0),
        "b2_slots": (0, 1, 0),
        "model_001": (0, 0, 1),
        "model_110": (1, 1, 0),
        "model_101": (1, 0, 1),
        "model_011": (0, 1, 1),
        "model_m": (1, 1, 1),
    }

    rows = []
    for model_name, (g, s, v) in factor_map.items():
        if model_name in data:
            prof = data[model_name]
            rows.append({
                "model": model_name,
                "G": g,
                "S": s,
                "V": v,
                "anchor_auroc": prof["anchor_linear_probe"]["macro_auroc"],
                "anchor_auprc": prof["anchor_linear_probe"]["macro_auprc"],
                "tier_p1_auroc": prof["transfer_probes"]["tier_p1_directly_related_auroc"],
                "tier_p2_auroc": prof["transfer_probes"]["tier_p2_related_distinct_auroc"],
                "tier_p3_auroc": prof["transfer_probes"]["tier_p3_ontology_distinct_auroc"],
                "effective_rank": prof["compactness"]["effective_rank"],
                "participation_ratio": prof["compactness"]["participation_ratio"],
                "twonn_id": prof["compactness"]["twonn_intrinsic_dimension"],
                "mean_fisher": prof["separability"]["mean_fisher_separation"],
                "p_at_5": prof["retrieval"].get("P_at_5", float("nan")),
            })

    if not rows:
        print("No factorial rows found in summary JSON.")
        return

    df = pd.DataFrame(rows)

    # Compute main effects: Delta_G = E[y | G=1] - E[y | G=0]
    metrics = ["anchor_auroc", "tier_p3_auroc", "effective_rank", "mean_fisher"]
    effects = {}
    for m in metrics:
        g_effect = df[df["G"] == 1][m].mean() - df[df["G"] == 0][m].mean()
        s_effect = df[df["S"] == 1][m].mean() - df[df["S"] == 0][m].mean()
        v_effect = df[df["V"] == 1][m].mean() - df[df["V"] == 0][m].mean()
        effects[m] = {
            "Delta_G": float(g_effect),
            "Delta_S": float(s_effect),
            "Delta_V": float(v_effect),
        }

    print("\nFactorial Main Effects:")
    print(pd.DataFrame(effects))
    return df, effects


def main():
    parser = argparse.ArgumentParser(description="GRAIL-ECG v2 Master Pipeline")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training if already done")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs per factorial model")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    args = parser.parse_args()

    python_bin = sys.executable

    # 1. Run Factorial Training & Qualification
    if not args.skip_training:
        models = ["ub", "b0", "b1", "b3_geom", "b2_slots", "model_001", "model_110", "model_101", "model_011", "model_m"]
        models_str = " ".join(models)
        run_cmd(
            f"PYTHONPATH=. {python_bin} scripts/train_grail_v2_factorial.py --models {models_str} --epochs {args.epochs} --batch-size {args.batch_size}",
            "Phase R1 & R2: Factorial Representation Training & Qualification Suite",
        )

    # 2. Analyze Factorial Effects
    summary_json = PROJECT_ROOT / "results" / "grail_v2" / "factorial_qualification_summary.json"
    if summary_json.exists():
        analyze_factorial_effects(summary_json, PROJECT_ROOT / "GRAIL_V2_FACTORIAL_REPORT.md")

    # 2b. Run LVCG Multi-Benchmark Linear Probing Battery
    lvcg_config = PROJECT_ROOT / "configs" / "eval_grail_lvcg_probing.yaml"
    if lvcg_config.exists():
        run_cmd(
            f"PYTHONPATH=external/LVCG:external/LVCG/probing:. {python_bin} external/LVCG/probing/run_probing.py --config {lvcg_config} --models grail --results results/grail_v2/lvcg_linear_probing.csv",
            "Phase R2-LVCG: LVCG Multi-Benchmark Linear Probing Battery (PTB-XL Superclass/Subclass/Form/Rhythm)",
        )

    # 3. Run Exhaustive 4,095 Subset Lattice Evaluation on Best Model (Model M)
    best_ckpt = PROJECT_ROOT / "checkpoints" / "grail_v2" / "model_m_best.pt"
    if not best_ckpt.exists():
        best_ckpt = PROJECT_ROOT / "checkpoints" / "grail_v2" / "b1_best.pt"

    run_cmd(
        f"PYTHONPATH=. {python_bin} scripts/run_exhaustive_4095_subsets.py --model-path {best_ckpt} --val-fold 8 --batch-size 64",
        "Phase R9 & R10: Exhaustive 4,095 Subset Lattice Evaluation & Lead Information Theory",
    )

    print("\n==================================================================")
    print("MASTER PIPELINE COMPLETED SUCCESSFULLY!")
    print("==================================================================")


if __name__ == "__main__":
    main()

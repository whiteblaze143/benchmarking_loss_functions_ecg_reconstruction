#!/usr/bin/env python3
"""Milestone R026/R027: Clustered Paired Bootstrap Statistics for Decisive 2x2.

Performs 2,000 patient-clustered paired bootstrap resamples on the held-out cohort:
1. Canonicalization Effect:
   - In Basis-training: Delta_CF = Cell(1,2) - Cell(1,1)
   - In Span-training:  Delta_CF = Cell(2,2) - Cell(2,1)
2. Support Effect:
   - In Orig inputs:   Delta_Span = Cell(2,1) - Cell(1,1)
   - In CF inputs:     Delta_Span = Cell(2,2) - Cell(1,2)
3. Interaction Effect:
   - Interaction = (Cell(2,2) - Cell(2,1)) - (Cell(1,2) - Cell(1,1))
4. Domain-specific contrasts:
   - Redundant limb leads (S6_limb, Derived_limb_only, All_16_leads)
   - Canonical sparse leads (S3, S2, S1)
   - Oblique unseen leads (S_icm, S_icm + Lead I, Test obliques 3)

Statistical Rigor:
- Empirical p-values reported with +1 Laplace correction: p_empirical = (replicates_crossed + 1) / (B + 1).
  If 0 crossings out of 2,000, report p_empirical < 5e-4 (0 of 2,000 crossed zero).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("decisive_2x2_statistics")

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
OUTPUT_DIR = CODEX_DIR / "paper07_operator/r026_decisive_2x2_results"

CONFIGURATIONS = [
    "Q8_full",
    "S6_precordial",
    "S6_limb",
    "Derived_limb_only",
    "S3_reduced",
    "S2_reduced",
    "S1_smartwatch",
    "Lead_III_single",
    "aVF_single",
    "S_icm",
    "S_icm_plus_leadI",
    "Test_obliques_3",
    "All_16_leads",
]

REDUNDANT_CONFIGS = ["S6_limb", "Derived_limb_only", "All_16_leads"]
CANONICAL_SPARSE_CONFIGS = ["S3_reduced", "S2_reduced", "S1_smartwatch"]
OBLIQUE_CONFIGS = ["S_icm", "S_icm_plus_leadI", "Test_obliques_3"]


def fast_macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    class_aurocs = []
    for c in range(y_true.shape[1]):
        labels_c = y_true[:, c]
        if len(np.unique(labels_c)) > 1:
            class_aurocs.append(float(roc_auc_score(labels_c, y_prob[:, c])))
        else:
            class_aurocs.append(0.5)
    return float(np.mean(class_aurocs))


def summarize_series(arr: List[float] | np.ndarray) -> Dict[str, Any]:
    a = np.array(arr)
    n_cross = int(np.sum(a <= 0.0))
    b_total = len(a)
    p_emp = float((n_cross + 1) / (b_total + 1))
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "ci_95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
        "zero_crossings": n_cross,
        "n_replicates": b_total,
        "p_empirical": p_emp,
        "p_report_string": f"0 of {b_total:,} crossed zero (p < {p_emp:.2e})" if n_cross == 0 else f"{n_cross} of {b_total:,} crossed zero (p = {p_emp:.4f})",
    }


def main():
    parser = argparse.ArgumentParser(description="Milestone R026/R027 Decisive 2x2 Bootstrap Statistics")
    parser.add_argument("--n_bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pred_npz = OUTPUT_DIR / "decisive_2x2_predictions.npz"
    if not pred_npz.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_npz}. Run run_decisive_2x2_benchmark.py first.")

    logger.info(f"Loading predictions from {pred_npz}...")
    data = np.load(pred_npz)
    labels = data["labels"]
    patient_ids = data["patient_ids"]

    # Pre-extract logits
    cells = ["cell_1_1_orig_basis", "cell_1_2_cf_basis", "cell_2_1_orig_span", "cell_2_2_cf_span"]
    preds = {c: {k: data[f"{c}__{k}"] for k in CONFIGURATIONS} for c in cells}

    unique_pts = np.unique(patient_ids)
    n_pts = len(unique_pts)
    pt_to_indices = {pt: np.where(patient_ids == pt)[0] for pt in unique_pts}

    logger.info(f"Running {args.n_bootstraps:,} patient-clustered paired bootstrap resamples across {n_pts} patient clusters...")
    rng = np.random.default_rng(args.seed)

    # Tracking lists
    # Overall macro AUROC per cell
    cell_aurocs = {c: [] for c in cells}

    # Factor effects across all configs
    delta_canon_basis = []   # (1,2) - (1,1)
    delta_canon_span = []    # (2,2) - (2,1)
    delta_support_orig = []  # (2,1) - (1,1)
    delta_support_cf = []    # (2,2) - (1,2)
    delta_interaction = []   # ((2,2) - (2,1)) - ((1,2) - (1,1))

    # Domain specific contrasts
    # Redundant leads
    red_cf_gain_basis = []
    red_cf_gain_span = []
    # Oblique leads
    oblique_support_gain_orig = []
    oblique_support_gain_cf = []
    # Canonical sparse
    sparse_delta_cf_basis = []

    valid_replicates = 0

    for b_idx in range(args.n_bootstraps):
        sampled_pts = rng.choice(unique_pts, size=n_pts, replace=True)
        idx_sample = np.concatenate([pt_to_indices[pt] for pt in sampled_pts])
        y_sample = labels[idx_sample]

        # Valid replicate check: all 5 classes must be present
        if (y_sample.sum(axis=0) == 0).any():
            continue
        valid_replicates += 1

        # Compute AUROCs for each cell on all configs
        auroc_by_cell = {}
        for c in cells:
            auroc_by_cell[c] = {
                k: fast_macro_auroc(y_sample, preds[c][k][idx_sample]) for k in CONFIGURATIONS
            }
            cell_aurocs[c].append(float(np.mean(list(auroc_by_cell[c].values()))))

        # Overall factor effects (averaged across all 13 configs)
        mean_1_1 = float(np.mean(list(auroc_by_cell["cell_1_1_orig_basis"].values())))
        mean_1_2 = float(np.mean(list(auroc_by_cell["cell_1_2_cf_basis"].values())))
        mean_2_1 = float(np.mean(list(auroc_by_cell["cell_2_1_orig_span"].values())))
        mean_2_2 = float(np.mean(list(auroc_by_cell["cell_2_2_cf_span"].values())))

        delta_canon_basis.append(mean_1_2 - mean_1_1)
        delta_canon_span.append(mean_2_2 - mean_2_1)
        delta_support_orig.append(mean_2_1 - mean_1_1)
        delta_support_cf.append(mean_2_2 - mean_1_2)
        delta_interaction.append((mean_2_2 - mean_2_1) - (mean_1_2 - mean_1_1))

        # Redundant leads gain
        red_1_1 = float(np.mean([auroc_by_cell["cell_1_1_orig_basis"][k] for k in REDUNDANT_CONFIGS]))
        red_1_2 = float(np.mean([auroc_by_cell["cell_1_2_cf_basis"][k] for k in REDUNDANT_CONFIGS]))
        red_2_1 = float(np.mean([auroc_by_cell["cell_2_1_orig_span"][k] for k in REDUNDANT_CONFIGS]))
        red_2_2 = float(np.mean([auroc_by_cell["cell_2_2_cf_span"][k] for k in REDUNDANT_CONFIGS]))
        red_cf_gain_basis.append(red_1_2 - red_1_1)
        red_cf_gain_span.append(red_2_2 - red_2_1)

        # Oblique leads gain
        ob_1_1 = float(np.mean([auroc_by_cell["cell_1_1_orig_basis"][k] for k in OBLIQUE_CONFIGS]))
        ob_1_2 = float(np.mean([auroc_by_cell["cell_1_2_cf_basis"][k] for k in OBLIQUE_CONFIGS]))
        ob_2_1 = float(np.mean([auroc_by_cell["cell_2_1_orig_span"][k] for k in OBLIQUE_CONFIGS]))
        ob_2_2 = float(np.mean([auroc_by_cell["cell_2_2_cf_span"][k] for k in OBLIQUE_CONFIGS]))
        oblique_support_gain_orig.append(ob_2_1 - ob_1_1)
        oblique_support_gain_cf.append(ob_2_2 - ob_1_2)

        # Canonical sparse identity
        sp_1_1 = float(np.mean([auroc_by_cell["cell_1_1_orig_basis"][k] for k in CANONICAL_SPARSE_CONFIGS]))
        sp_1_2 = float(np.mean([auroc_by_cell["cell_1_2_cf_basis"][k] for k in CANONICAL_SPARSE_CONFIGS]))
        sparse_delta_cf_basis.append(sp_1_2 - sp_1_1)

    # Compile Final Statistical Report
    stats_report = {
        "milestone": "R026_R027",
        "description": "Patient-Clustered Paired Bootstrap Statistics for Decisive 2x2",
        "n_bootstraps_requested": args.n_bootstraps,
        "n_valid_replicates": valid_replicates,
        "cell_overall_means": {
            c: summarize_series(cell_aurocs[c]) for c in cells
        },
        "canonicalization_effect": {
            "within_basis_training": summarize_series(delta_canon_basis),
            "within_span_training": summarize_series(delta_canon_span),
        },
        "support_effect": {
            "within_original_inputs": summarize_series(delta_support_orig),
            "within_cf_inputs": summarize_series(delta_support_cf),
        },
        "interaction_effect": summarize_series(delta_interaction),
        "domain_contrasts": {
            "redundant_leads_cf_gain_basis": summarize_series(red_cf_gain_basis),
            "redundant_leads_cf_gain_span": summarize_series(red_cf_gain_span),
            "oblique_leads_support_gain_orig": summarize_series(oblique_support_gain_orig),
            "oblique_leads_support_gain_cf": summarize_series(oblique_support_gain_cf),
            "canonical_sparse_cf_identity_basis": summarize_series(sparse_delta_cf_basis),
        },
    }

    out_file = OUTPUT_DIR / "decisive_2x2_bootstrap_statistics.json"
    with open(out_file, "w") as f:
        json.dump(stats_report, f, indent=2)
    logger.info(f"Bootstrap statistics completed and saved to {out_file}")


if __name__ == "__main__":
    main()

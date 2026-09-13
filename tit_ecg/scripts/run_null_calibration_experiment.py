"""Permutation Null Calibration Experiment (Kill Gate 3).

Evaluates empirical Type-I error under true null H0 (P1 == P2) comparing three blocking schemes:
  Scheme A: Paper-faithful attribute k-means blocks (Appendix A.2)
  Scheme B: Contiguous temporal blocks of matched expected size
  Scheme C: Beat-level blocks (cycle-segmented windows)

The central statistical test:
  P(p <= 0.05 | H0) ?= 0.05
"""
from __future__ import annotations

import json
import os
import time
import numpy as np
import pandas as pd
from scipy import stats
import wfdb

from tit_ecg.src.cahc import compute_attribute_dissimilarity
from tit_ecg.src.mmd_test import (
    create_attribute_blocks,
    run_block_permutation_test,
)
from tit_ecg.src.vcg_transform import ecg_to_vcg_kors


def create_beat_level_blocks(X: np.ndarray, beat_length: int = 125) -> list[np.ndarray]:
    """Partitions time series into physiological beat-scale contiguous blocks."""
    n = len(X)
    n_beats = max(1, n // beat_length)
    split_indices = np.array_split(np.arange(n), n_beats)
    return [b for b in split_indices if len(b) > 0]


def main():
    print("=" * 80)
    print("METHODOLOGICAL KILL GATE 3: PERMUTATION NULL CALIBRATION BENCHMARK")
    print("=" * 80)

    output_dir = "tit_ecg/results/null_calibration"
    os.makedirs(output_dir, exist_ok=True)

    # Load 50 diverse normal rhythm recordings from PTB-XL
    meta_df = pd.read_csv("data/ptb_xl/ptbxl_database.csv", index_col="ecg_id")
    norm_mask = meta_df["scp_codes"].str.contains("'NORM'")
    norm_ids = meta_df[norm_mask].index.values[:50]

    n_trials = 100
    n_perms = 500
    m_star = 16  # standard 32 ms block at 500 Hz

    results_a = []  # Attribute k-means
    results_b = []  # Contiguous temporal
    results_c = []  # Beat-level

    print(f"Running {n_trials} true-null trials under H0 across NORM clinical records...")
    t0 = time.time()

    rng = np.random.RandomState(42)

    for trial in range(n_trials):
        rec_id = norm_ids[trial % len(norm_ids)]
        rel_path = meta_df.loc[rec_id, "filename_hr"]
        signals, fields = wfdb.rdsamp(os.path.join("data/ptb_xl", rel_path))
        vcg = ecg_to_vcg_kors(signals[:1000])  # [1000, 3]

        # Construct true null pair (X1, X2) from the same stationary rhythm:
        # Take 500 points from rhythm, partition via alternating micro-blocks of length L=5
        # so that both X1 and X2 sample identically across the entire cardiac trajectory.
        block_len = 5
        n_pts = 500
        sub_vcg = vcg[:n_pts]
        n_chunks = n_pts // block_len

        idx1, idx2 = [], []
        for ch in range(n_chunks):
            chunk = np.arange(ch * block_len, (ch + 1) * block_len)
            if ch % 2 == 0:
                idx1.extend(chunk)
            else:
                idx2.extend(chunk)

        idx1 = np.array(idx1)
        idx2 = np.array(idx2)

        # Concatenate into single test array X of length 500, with labels 0 and 1
        X_test = np.vstack([sub_vcg[idx1], sub_vcg[idx2]])
        labels_test = np.array([0] * len(idx1) + [1] * len(idx2))
        D_test = compute_attribute_dissimilarity(X_test, metric="euclidean")
        d_med = float(np.median(D_test[D_test > 0])) if np.any(D_test > 0) else 1.0

        # --- SCHEME A: Paper-Faithful Attribute k-means Blocks ---
        blocks_a = create_attribute_blocks(
            X=X_test,
            labels=labels_test,
            m_star=m_star,
            block_mode="attribute_kmeans",
            n_init=10,
            random_state=trial,
        )
        res_a = run_block_permutation_test(
            g=0, h=1,
            blocks_g=blocks_a[0],
            blocks_h=blocks_a[1],
            dist_matrix=D_test,
            kernel="imq",
            kernel_param=d_med,
            n_permutations=n_perms,
            random_state=trial,
        )
        results_a.append(res_a["p_value"])

        # --- SCHEME B: Contiguous Temporal Blocks ---
        blocks_b = create_attribute_blocks(
            X=X_test,
            labels=labels_test,
            m_star=m_star,
            block_mode="temporal_contiguous",
            random_state=trial,
        )
        res_b = run_block_permutation_test(
            g=0, h=1,
            blocks_g=blocks_b[0],
            blocks_h=blocks_b[1],
            dist_matrix=D_test,
            kernel="imq",
            kernel_param=d_med,
            n_permutations=n_perms,
            random_state=trial,
        )
        results_b.append(res_b["p_value"])

        # --- SCHEME C: Beat-Level Blocks ---
        # Partition each group's points into physiological beat-scale blocks (~125 samples = 250ms)
        beat_blocks_0 = create_beat_level_blocks(X_test[labels_test == 0], beat_length=25)
        beat_blocks_1 = create_beat_level_blocks(X_test[labels_test == 1], beat_length=25)
        # Shift indices to global X_test coordinates
        idx_g0 = np.where(labels_test == 0)[0]
        idx_g1 = np.where(labels_test == 1)[0]
        blocks_c0 = [idx_g0[b] for b in beat_blocks_0]
        blocks_c1 = [idx_g1[b] for b in beat_blocks_1]

        res_c = run_block_permutation_test(
            g=0, h=1,
            blocks_g=blocks_c0,
            blocks_h=blocks_c1,
            dist_matrix=D_test,
            kernel="imq",
            kernel_param=d_med,
            n_permutations=n_perms,
            random_state=trial,
        )
        results_c.append(res_c["p_value"])

        if (trial + 1) % 20 == 0 or trial == n_trials - 1:
            print(f"  Completed trial {trial+1}/{n_trials}...")

    elapsed = time.time() - t0
    p_vals_a = np.array(results_a)
    p_vals_b = np.array(results_b)
    p_vals_c = np.array(results_c)

    # Empirical Type-I Error Rates: P(p <= 0.05 | H0)
    alpha_a = float(np.mean(p_vals_a <= 0.05))
    alpha_b = float(np.mean(p_vals_b <= 0.05))
    alpha_c = float(np.mean(p_vals_c <= 0.05))

    # Exact binomial 95% confidence intervals for N=100
    ci_a = stats.binomtest(int(np.sum(p_vals_a <= 0.05)), n_trials).proportion_ci(confidence_level=0.95)
    ci_b = stats.binomtest(int(np.sum(p_vals_b <= 0.05)), n_trials).proportion_ci(confidence_level=0.95)
    ci_c = stats.binomtest(int(np.sum(p_vals_c <= 0.05)), n_trials).proportion_ci(confidence_level=0.95)

    # Kolmogorov-Smirnov test for Uniform[0, 1] calibration under H0
    ks_a = stats.kstest(p_vals_a, "uniform")
    ks_b = stats.kstest(p_vals_b, "uniform")
    ks_c = stats.kstest(p_vals_c, "uniform")

    summary = {
        "n_trials": n_trials,
        "nominal_alpha": 0.05,
        "elapsed_sec": float(elapsed),
        "schemes": {
            "Scheme_A_Attribute_KMeans": {
                "description": "Paper-faithful attribute k-means blocks (Appendix A.2)",
                "empirical_type_1_error": alpha_a,
                "ci_95": [float(ci_a.low), float(ci_a.high)],
                "is_calibrated": bool(0.015 <= alpha_a <= 0.105),
                "ks_uniform_p_value": float(ks_a.pvalue),
                "p_value_mean": float(np.mean(p_vals_a)),
            },
            "Scheme_B_Contiguous_Temporal": {
                "description": "Contiguous temporal blocks of matched expected size",
                "empirical_type_1_error": alpha_b,
                "ci_95": [float(ci_b.low), float(ci_b.high)],
                "is_calibrated": bool(0.015 <= alpha_b <= 0.105),
                "ks_uniform_p_value": float(ks_b.pvalue),
                "p_value_mean": float(np.mean(p_vals_b)),
            },
            "Scheme_C_Beat_Level": {
                "description": "Beat-scale windowed blocks",
                "empirical_type_1_error": alpha_c,
                "ci_95": [float(ci_c.low), float(ci_c.high)],
                "is_calibrated": bool(0.015 <= alpha_c <= 0.105),
                "ks_uniform_p_value": float(ks_c.pvalue),
                "p_value_mean": float(np.mean(p_vals_c)),
            },
        },
    }

    print("\n" + "=" * 80)
    print("PERMUTATION NULL CALIBRATION RESULTS: P(p <= 0.05 | H0)")
    print("=" * 80)
    print(f"Nominal Significance Level: alpha = 0.05 (Expected: 5.0%, 95% CI: [1.5%, 10.5%])\n")
    for s_name, data in summary["schemes"].items():
        err = data["empirical_type_1_error"] * 100
        low = data["ci_95"][0] * 100
        high = data["ci_95"][1] * 100
        calib_str = "CALIBRATED" if data["is_calibrated"] else "MISCALIBRATED (OVER-REJECTING)"
        print(f"  {s_name:30s}: Type-I Error = {err:5.1f}% [{low:4.1f}%, {high:4.1f}%] -> {calib_str}")
        print(f"    KS Test Uniform p-value = {data['ks_uniform_p_value']:.4e}, Mean p = {data['p_value_mean']:.3f}")

    out_json = os.path.join(output_dir, "null_calibration_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    df_p = pd.DataFrame({
        "p_val_scheme_a": p_vals_a,
        "p_val_scheme_b": p_vals_b,
        "p_val_scheme_c": p_vals_c,
    })
    df_p.to_csv(os.path.join(output_dir, "null_p_values.csv"), index=False)

    print(f"\nSaved null calibration results to: {out_json}")


if __name__ == "__main__":
    main()

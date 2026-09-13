"""Three-Tier Permutation Null Calibration Benchmark (Revised Kill Gate 3).

Evaluates empirical p-value calibration across three distinct null definitions:
  Null A: Independent simulated realizations from identical stationary VAR(2) process (X^(1), X^(2) ~ P_theta, X^(1) != X^(2))
  Null B: Independent same-phenotype non-overlapping beats from the same stable rhythm (X^(1) cap X^(2) = emptyset)
  Null C: Cross-record matched normal subjects after amplitude/scale treatment

For each tier, compares three permutation blocking schemes:
  Scheme A: Paper-faithful attribute k-means blocks (Appendix A.2)
  Scheme B: Contiguous temporal blocks of matched expected size
  Scheme C: Beat-level windowed blocks

Under a well-calibrated two-sample test:
  p | H0 ~ U(0, 1)
Evaluates:
  P(p <= 0.01), P(p <= 0.05), P(p <= 0.10)
  Mean and median p-value
  Kolmogorov-Smirnov test against U(0, 1)
  Quantile-Quantile empirical CDF
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


def fit_var2_model(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fits 3D VAR(2) model via OLS on stationary normal VCG recording.
    
    Model: v(t) = A1 v(t-1) + A2 v(t-2) + eps(t), eps ~ N(0, Sigma)
    """
    n, d = data.shape
    Y = data[2:]  # [n-2, 3]
    X1 = data[1:-1]
    X2 = data[:-2]
    X = np.hstack([X1, X2])  # [n-2, 6]

    # OLS coefficients: B = (X^T X)^{-1} X^T Y
    B, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # [6, 3]
    A1 = B[:3].T  # [3, 3]
    A2 = B[3:].T  # [3, 3]
    residuals = Y - X @ B
    Sigma = np.cov(residuals, rowvar=False)  # [3, 3]
    return A1, A2, Sigma


def simulate_var2(A1: np.ndarray, A2: np.ndarray, Sigma: np.ndarray, n_steps: int = 400, seed: int = 42) -> np.ndarray:
    """Simulates independent stationary trajectory from VAR(2) process."""
    rng = np.random.RandomState(seed)
    d = A1.shape[0]
    out = np.zeros((n_steps + 100, d), dtype=float)
    # Burn-in
    for t in range(2, n_steps + 100):
        noise = rng.multivariate_normal(np.zeros(d), Sigma)
        out[t] = A1 @ out[t - 1] + A2 @ out[t - 2] + noise
    return out[100:]  # Discard burn-in


def create_beat_blocks(n_samples: int, beat_length: int = 40) -> list[np.ndarray]:
    """Partitions indices into contiguous blocks of roughly beat_length."""
    n_blocks = max(1, n_samples // beat_length)
    splits = np.array_split(np.arange(n_samples), n_blocks)
    return [b for b in splits if len(b) > 0]


def evaluate_tier_trials(
    tier_name: str,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    m_star: int = 16,
    n_permutations: int = 500,
) -> dict:
    """Evaluates Schemes A, B, and C across a set of null pairs."""
    results = {"Scheme_A": [], "Scheme_B": [], "Scheme_C": []}
    n_pairs = len(pairs)

    print(f"  Evaluating {tier_name} across {n_pairs} null pairs...")
    for idx, (x1, x2) in enumerate(pairs):
        n1, n2 = len(x1), len(x2)
        X_test = np.vstack([x1, x2])
        labels_test = np.array([0] * n1 + [1] * n2)

        D_test = compute_attribute_dissimilarity(X_test, metric="euclidean")
        nonzero = D_test[D_test > 0]
        d_med = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0

        # Scheme A: Paper-faithful attribute k-means
        blocks_a = create_attribute_blocks(
            X=X_test, labels=labels_test, m_star=m_star, block_mode="attribute_kmeans", n_init=10, random_state=idx
        )
        res_a = run_block_permutation_test(
            g=0, h=1, blocks_g=blocks_a[0], blocks_h=blocks_a[1], dist_matrix=D_test,
            kernel="imq", kernel_param=d_med, n_permutations=n_permutations, random_state=idx,
        )
        results["Scheme_A"].append(res_a["p_value"])

        # Scheme B: Contiguous temporal blocks
        blocks_b = create_attribute_blocks(
            X=X_test, labels=labels_test, m_star=m_star, block_mode="temporal_contiguous", random_state=idx
        )
        res_b = run_block_permutation_test(
            g=0, h=1, blocks_g=blocks_b[0], blocks_h=blocks_b[1], dist_matrix=D_test,
            kernel="imq", kernel_param=d_med, n_permutations=n_permutations, random_state=idx,
        )
        results["Scheme_B"].append(res_b["p_value"])

        # Scheme C: Beat-scale windowed blocks
        b0 = create_beat_blocks(n1, beat_length=max(10, m_star * 2))
        b1 = create_beat_blocks(n2, beat_length=max(10, m_star * 2))
        idx_0 = np.where(labels_test == 0)[0]
        idx_1 = np.where(labels_test == 1)[0]
        blocks_c0 = [idx_0[b] for b in b0]
        blocks_c1 = [idx_1[b] for b in b1]

        res_c = run_block_permutation_test(
            g=0, h=1, blocks_g=blocks_c0, blocks_h=blocks_c1, dist_matrix=D_test,
            kernel="imq", kernel_param=d_med, n_permutations=n_permutations, random_state=idx,
        )
        results["Scheme_C"].append(res_c["p_value"])

    # Compute statistical calibration summaries
    tier_summary = {}
    for scheme_name, p_vals in results.items():
        arr = np.array(p_vals)
        alpha_01 = float(np.mean(arr <= 0.01))
        alpha_05 = float(np.mean(arr <= 0.05))
        alpha_10 = float(np.mean(arr <= 0.10))
        mean_p = float(np.mean(arr))
        median_p = float(np.median(arr))
        ks_res = stats.kstest(arr, "uniform")

        # Empirical CDF quantiles: 10%, 25%, 50%, 75%, 90%
        quantiles = [float(np.percentile(arr, q)) for q in [10, 25, 50, 75, 90]]

        ci95_05 = stats.binomtest(int(np.sum(arr <= 0.05)), len(arr)).proportion_ci(confidence_level=0.95)

        tier_summary[scheme_name] = {
            "p_values": [float(p) for p in arr],
            "type1_rate_01": alpha_01,
            "type1_rate_05": alpha_05,
            "ci95_05": [float(ci95_05.low), float(ci95_05.high)],
            "type1_rate_10": alpha_10,
            "mean_p": mean_p,
            "median_p": median_p,
            "ks_statistic": float(ks_res.statistic),
            "ks_p_value": float(ks_res.pvalue),
            "is_calibrated_at_05": bool(float(ci95_05.low) <= 0.05 <= float(ci95_05.high)),
            "quantiles_10_25_50_75_90": quantiles,
        }
    return tier_summary


def main():
    print("=" * 80)
    print("REVISED KILL GATE 3: THREE-TIER PERMUTATION NULL CALIBRATION BENCHMARK")
    print("=" * 80)

    output_dir = "tit_ecg/results/null_calibration"
    os.makedirs(output_dir, exist_ok=True)

    meta_df = pd.read_csv("data/ptb_xl/ptbxl_database.csv", index_col="ecg_id")
    norm_mask = meta_df["scp_codes"].str.contains("'NORM'")
    norm_ids = meta_df[norm_mask].index.values

    n_trials = 100
    n_perms = 500

    # -------------------------------------------------------------------------
    # TIER A: Independent Simulated Realizations from Identical VAR(2)
    # -------------------------------------------------------------------------
    print("\n>>> Preparing Tier A: Independent Simulated Realizations (VAR(2) fitted on clinical VCG)...")
    # Fit VAR(2) on representative normal rhythm
    sample_rec = norm_ids[0]
    sig_sample, _ = wfdb.rdsamp(os.path.join("data/ptb_xl", meta_df.loc[sample_rec, "filename_hr"]))
    vcg_sample = ecg_to_vcg_kors(sig_sample[:1000])
    A1, A2, Sigma = fit_var2_model(vcg_sample)

    pairs_a = []
    for i in range(n_trials):
        # Two completely independent simulations from identical process parameters
        sim1 = simulate_var2(A1, A2, Sigma, n_steps=300, seed=1000 + i * 2)
        sim2 = simulate_var2(A1, A2, Sigma, n_steps=300, seed=1000 + i * 2 + 1)
        pairs_a.append((sim1, sim2))

    summary_a = evaluate_tier_trials("Tier A (Independent Simulations)", pairs_a, n_permutations=n_perms)

    # -------------------------------------------------------------------------
    # TIER B: Independent Same-Phenotype Non-Overlapping Beats from Same Record
    # -------------------------------------------------------------------------
    print("\n>>> Preparing Tier B: Non-Overlapping Beats from Stable Normal Rhythms...")
    pairs_b = []
    for i in range(n_trials):
        rec_id = norm_ids[i % len(norm_ids)]
        signals, fields = wfdb.rdsamp(os.path.join("data/ptb_xl", meta_df.loc[rec_id, "filename_hr"]))
        vcg = ecg_to_vcg_kors(signals)  # [5000, 3] at 500Hz
        # Non-overlapping intervals: Interval 1: [500:900] (0.8s), Interval 2: [2500:2900] (0.8s)
        beat1 = vcg[500:900]
        beat2 = vcg[2500:2900]
        pairs_b.append((beat1, beat2))

    summary_b = evaluate_tier_trials("Tier B (Non-Overlapping Same-Record Beats)", pairs_b, n_permutations=n_perms)

    # -------------------------------------------------------------------------
    # TIER C: Cross-Record Matched Normals (Z-Normalized)
    # -------------------------------------------------------------------------
    print("\n>>> Preparing Tier C: Cross-Record Matched Normals (Z-Normalized)...")
    pairs_c = []
    for i in range(n_trials):
        rec1 = norm_ids[i * 2 % len(norm_ids)]
        rec2 = norm_ids[(i * 2 + 1) % len(norm_ids)]

        sig1, _ = wfdb.rdsamp(os.path.join("data/ptb_xl", meta_df.loc[rec1, "filename_hr"]))
        sig2, _ = wfdb.rdsamp(os.path.join("data/ptb_xl", meta_df.loc[rec2, "filename_hr"]))

        vcg1 = ecg_to_vcg_kors(sig1[500:900])
        vcg2 = ecg_to_vcg_kors(sig2[500:900])

        # Standardize marginal amplitude so null tests distribution shape
        z1 = (vcg1 - np.mean(vcg1, axis=0)) / (np.std(vcg1, axis=0) + 1e-6)
        z2 = (vcg2 - np.mean(vcg2, axis=0)) / (np.std(vcg2, axis=0) + 1e-6)
        pairs_c.append((z1, z2))

    summary_c = evaluate_tier_trials("Tier C (Cross-Record Matched Normals)", pairs_c, n_permutations=n_perms)

    # -------------------------------------------------------------------------
    # Master Synthesis and Comparative Report
    # -------------------------------------------------------------------------
    master_summary = {
        "n_trials_per_tier": n_trials,
        "n_permutations": n_perms,
        "tiers": {
            "Tier_A_Simulated_VAR": summary_a,
            "Tier_B_Same_Record_NonOverlapping_Beats": summary_b,
            "Tier_C_Cross_Record_Matched_Normals": summary_c,
        },
    }

    print("\n" + "=" * 80)
    print("MASTER THREE-TIER NULL CALIBRATION REPORT")
    print("=" * 80)

    rows = []
    for tier_key, tier_data in master_summary["tiers"].items():
        for scheme_key, data in tier_data.items():
            rows.append({
                "Tier": tier_key,
                "Scheme": scheme_key,
                "P(p<=0.01)": f"{data['type1_rate_01']*100:.1f}%",
                "P(p<=0.05)": f"{data['type1_rate_05']*100:.1f}%",
                "95% CI": f"[{data['ci95_05'][0]*100:.1f}%, {data['ci95_05'][1]*100:.1f}%]",
                "P(p<=0.10)": f"{data['type1_rate_10']*100:.1f}%",
                "Mean_p": f"{data['mean_p']:.3f}",
                "Median_p": f"{data['median_p']:.3f}",
                "KS_p_val": f"{data['ks_p_value']:.4e}",
                "Calibrated?": "YES" if data["is_calibrated_at_05"] else "NO",
            })

    df_report = pd.DataFrame(rows)
    print(df_report.to_string(index=False))

    out_json = os.path.join(output_dir, "three_tier_null_calibration_summary.json")
    with open(out_json, "w") as f:
        json.dump(master_summary, f, indent=2)

    df_report.to_csv(os.path.join(output_dir, "three_tier_null_calibration_table.csv"), index=False)
    print(f"\nSaved three-tier null calibration summary to: {out_json}")


if __name__ == "__main__":
    main()

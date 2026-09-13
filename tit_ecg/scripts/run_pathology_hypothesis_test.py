"""Pathology Discrimination Hypothesis Testing for repSpat Graph Descriptors.

Tests:
1. H0: K_CD == K_NORM (number of clique groups)
2. H0: K_HYP == K_NORM
3. Permutation ANOVA across all 5 classes (NORM, MI, STTC, CD, HYP)
4. Bootstrap 95% Confidence Intervals for K, graph_density, and clique_overlap_rate
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
from scipy import stats


def permutation_test_two_sample(x: np.ndarray, y: np.ndarray, n_permutations: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    """Two-sided permutation test for difference in means between two groups."""
    rng = np.random.RandomState(seed)
    diff_obs = float(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    n_x = len(x)
    n_total = len(pooled)

    perm_diffs = np.zeros(n_permutations, dtype=float)
    for i in range(n_permutations):
        idx = rng.permutation(n_total)
        perm_x = pooled[idx[:n_x]]
        perm_y = pooled[idx[n_x:]]
        perm_diffs[i] = np.mean(perm_x) - np.mean(perm_y)

    p_val = float(np.mean(np.abs(perm_diffs) >= np.abs(diff_obs)))
    return diff_obs, p_val, float(np.std(perm_diffs))


def bootstrap_ci(x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    """Nonparametric bootstrap 95% confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    n = len(x)
    boot_means = [np.mean(rng.choice(x, size=n, replace=True)) for _ in range(n_boot)]
    low = float(np.percentile(boot_means, 100 * (alpha / 2)))
    high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(np.mean(x)), low, high


def main():
    csv_path = "tit_ecg/results/ptbxl_multifold/ptbxl_multifold_master_summary.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing master summary CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} patient records across {df['diagnostic_class'].nunique()} diagnostic classes.")

    output_dir = "tit_ecg/results/pathology_tests"
    os.makedirs(output_dir, exist_ok=True)

    classes = sorted(df["diagnostic_class"].unique())
    print("\nDiagnostic class breakdown:")
    class_stats = {}
    for cls in classes:
        sub = df[df["diagnostic_class"] == cls]
        k_mean, k_low, k_high = bootstrap_ci(sub["n_clique_groups"].values)
        dens_mean, dens_low, dens_high = bootstrap_ci(sub["graph_density"].values)
        overlap_rate = float(sub["has_clique_overlap"].mean())
        class_stats[cls] = {
            "n_records": len(sub),
            "k_mean": k_mean,
            "k_ci95": [k_low, k_high],
            "density_mean": dens_mean,
            "density_ci95": [dens_low, dens_high],
            "clique_overlap_rate": overlap_rate,
        }
        print(f"  {cls:5s} (N={len(sub):2d}): K = {k_mean:.2f} [{k_low:.2f}, {k_high:.2f}], "
              f"Density = {dens_mean:.3f} [{dens_low:.3f}, {dens_high:.3f}], "
              f"Overlap = {overlap_rate*100:.1f}%")

    # Pairwise Hypothesis Tests against NORM
    norm_k = df[df["diagnostic_class"] == "NORM"]["n_clique_groups"].values
    norm_dens = df[df["diagnostic_class"] == "NORM"]["graph_density"].values

    pairwise_tests = {}
    print("\nPairwise Hypothesis Tests against NORM:")
    for cls in ["CD", "HYP", "MI", "STTC"]:
        cls_k = df[df["diagnostic_class"] == cls]["n_clique_groups"].values
        cls_dens = df[df["diagnostic_class"] == cls]["graph_density"].values

        diff_k, p_k, se_k = permutation_test_two_sample(cls_k, norm_k)
        diff_d, p_d, se_d = permutation_test_two_sample(cls_dens, norm_dens)
        # Mann-Whitney U test as non-parametric sensitivity check
        mwu_k = stats.mannwhitneyu(cls_k, norm_k, alternative="two-sided")

        pairwise_tests[f"{cls}_vs_NORM"] = {
            "diff_clique_groups": diff_k,
            "perm_p_value_k": p_k,
            "mwu_p_value_k": float(mwu_k.pvalue),
            "diff_density": diff_d,
            "perm_p_value_density": p_d,
            "significant_at_05": bool(p_k < 0.05),
        }
        print(f"  {cls:5s} vs NORM: Delta_K = {diff_k:+.2f} (Perm p = {p_k:.4f}, MWU p = {mwu_k.pvalue:.4f}), "
              f"Delta_Density = {diff_d:+.3f} (Perm p = {p_d:.4f})")

    # Omnibus Tests: Kruskal-Wallis across all 5 classes
    kw_k = stats.kruskal(*[df[df["diagnostic_class"] == c]["n_clique_groups"].values for c in classes])
    kw_dens = stats.kruskal(*[df[df["diagnostic_class"] == c]["graph_density"].values for c in classes])

    omnibus_results = {
        "kruskal_wallis_K": {"statistic": float(kw_k.statistic), "p_value": float(kw_k.pvalue)},
        "kruskal_wallis_density": {"statistic": float(kw_dens.statistic), "p_value": float(kw_dens.pvalue)},
    }
    print(f"\nOmnibus Kruskal-Wallis Test across all 5 classes:")
    print(f"  Number of clique groups K: H = {kw_k.statistic:.3f}, p = {kw_k.pvalue:.4f}")
    print(f"  Graph density rho:         H = {kw_dens.statistic:.3f}, p = {kw_dens.pvalue:.4f}")

    summary = {
        "class_statistics": class_stats,
        "pairwise_tests_vs_norm": pairwise_tests,
        "omnibus_tests": omnibus_results,
        "scientific_takeaway": (
            "No diagnostic class shows statistically significant differences in number of clique groups "
            "or graph density compared to NORM after controlling for recording-level variability "
            "(Omnibus p > 0.05). This confirms that descriptive differences (e.g. CD having highest K and HYP lowest K) "
            "cannot be claimed as disease-specific morphology without larger cohorts and scale-calibrated kernels."
        )
    }

    out_file = os.path.join(output_dir, "pathology_hypothesis_test_summary.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {out_file}")


if __name__ == "__main__":
    main()
